## adversarial loss in evaluation
## add model for discriminator
## change the parameters appropriately


import torch
import os
import argparse
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset import wav_mel_dataset, get_dataset
from model.generator import Generator
from model.vocoder_discriminator import MultiScaleDiscriminator, MultiBandDiscriminator
from model.util import load_try, build_mel_basis, tr_amp_to_db, tr_normalize, tr_pre


from config import Config
from utils.utils import load_checkpoint, scan_checkpoint, save_checkpoint, build_env, AttrDict, get_mel_pytorch, load_config

from losses import timeLoss, specLoss

class Vocoder(torch.nn.Module):
    def __init__(self, sample_rate):
        super(Vocoder, self).__init__()
        Config.refresh(sample_rate)
        self.rate = sample_rate
        if(not os.path.exists(Config.ckpt)):
            raise RuntimeError("Model not found. Please download the model")
        self._load_pretrain(Config.ckpt)
        self.weight_torch = Config.get_mel_weight_torch(percent=1.0)[None, None, None, ...]

    def _load_pretrain(self, pth):
        self.model = Generator(Config.cin_channels)
        checkpoint = load_checkpoint(pth, torch.device("cpu"))
        load_try(checkpoint["generator"], self.model)
        
    def forward(self, mel, cuda=False):
        """
        :param non normalized mel spectrogram: [batchsize, 1, t-steps, n_mel]
        :return: [batchsize, 1, samples]
        """
        assert mel.size()[-1] == 128
        # self.weight_torch = self.weight_torch.type_as(mel)
        # mel = mel / self.weight_torch
        mel = tr_normalize(tr_amp_to_db(torch.abs(mel)) - 20.0)
        mel = tr_pre(mel[:, 0, ...])
        wav_re = self.model(mel)
        return wav_re



def train(config):
    torch.cuda.manual_seed(config.seed)
    device = torch.device("cuda")
    
    # load model
    model = Vocoder(config.sample_rate)
    model = model.to(device)

    ## load pre-trained model here
    
    if os.path.isdir(config.checkpoint_path):
        cp_voc = scan_checkpoint(config.checkpoint_path, 'voc_')

    # load last checkpoint if exists
    steps = 0
    if cp_voc is None:
        state_dict = None
        last_epoch = -1
    else:
        state_dict = load_checkpoint(cp_voc, device)
        print(f"Loading state dict from {cp_voc}...")
        model.load_state_dict(state_dict['model'])
        try:
            steps = state_dict['steps'] + 1
            last_epoch = state_dict['epoch']
        except:
            steps = int(input(f"Enter steps for {cp_voc}"))
            last_epoch = int(input(f"Enter epochs for {cp_voc}"))
    
    # optimizer

    optim = torch.optim.Adam(model.parameters(), config.learning_rate, betas=[config.adam_b1, config.adam_b2])

    if state_dict is not None:
        try:
            optim.load_state_dict(state_dict['optim'])
        except:
            print("Warning: Did not find optim state in checkpoint!")
            pass

    
    # dataset and dataloader
    ## train dataset and dataloader
    train_paths = get_dataset(config.data_path, 'train')
    train_dataset = wav_mel_dataset(config, train_paths)

    ## val dataset and dataloader
    val_paths = get_dataset(config.data_path, 'val')
    val_dataset = wav_mel_dataset(config, val_paths, mode='val')

    qual_sample_idcs = np.random.randint(len(val_dataset), size=(config.num_qual_samples,))

    train_loader = DataLoader(train_dataset, num_workers=0, shuffle=True, 
                                batch_size=config.batch_size, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=0, shuffle=False, 
                                batch_size=config.batch_size, pin_memory=True, drop_last=False)

    sw = SummaryWriter(os.path.join(config.checkpoint_path, 'logs'))

    model.train()


    ## loss function
    losses_std_avg = 0.0
    losses_tb_avg = 0.0
    loss_mel_tb_avg, loss_spec_tb_avg, loss_time_tb_avg = 0.0, 0.0, 0.0
    std_ctr = 0
    tb_ctr = 0
    adv_loss, loss_adv_tb_avg = 0.0, 0.0
    loss_d, loss_dsc_tb_avg = 0.0, 0.0 

    mel_recon_loss = torch.nn.MSELoss(reduction='mean')
    spec_loss_fun = specLoss(lam_mag = config.lam_mag, lam_sc = config.lam_sc)
    time_loss_fun = timeLoss(lam_seg = config.lam_seg, 
                lam_energy = config.lam_energy,
                lam_phase = config.lam_phase)

    window = torch.hann_window(Config.n_fft).to(device)
    mel_basis = torch.from_numpy(build_mel_basis()).to(device)

    for epoch in range(max(0, last_epoch), config.train_epochs):
        start = time.time()
        print(f"Epoch: {epoch}")

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            inp_time = batch['wav'].to(device).float()

            ## convert the input time signal to melspectrogram
            inp_mel = get_mel_pytorch(inp_time, Config.n_fft, Config.hop_length, 
                                        Config.win_size,window, mel_basis)

            pred_time = model(inp_mel, cuda=True)

            pred_time = pred_time.squeeze(1)
            if pred_time.size(-1) > inp_time.size(-1):
                pred_time = pred_time[:, :inp_time.size(-1)]

            ## check here the extra dimension/ unsqueeze
            pred_mel = get_mel_pytorch(pred_time, Config.n_fft, Config.hop_length,
                                        Config.win_size, window, mel_basis)

            optim.zero_grad()

            ## multiple losses
            loss_mel = mel_recon_loss(pred_mel, inp_mel)
            loss_spec = spec_loss_fun(pred_time, inp_time)
            loss_time = time_loss_fun(pred_time, inp_time)
            
            loss = config.lam_mel*loss_mel + loss_spec + loss_time

            if epoch > opt.discriminator_train_start_epoch:
                # generator
                disc_fake, disc_fake_subband, disc_fake_freq = model_dsc(pred_time.unsqueeze(1))
                disc_real, disc_real_subband, disc_real_freq = model_dsc(gt_time.unsqueeze(1))

                adv_loss_multi_scale = 0.0
                for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                    adv_loss_multi_scale += criterion(score_fake, torch.ones_like(score_fake))
                adv_loss_multi_scale /= len(disc_fake)
                
                adv_loss_mulit_band = 0.0
                for (_, score_fake), (_, score_real) in zip(disc_fake_subband, disc_real_subband):
                    adv_loss_mulit_band += criterion(score_fake, torch.ones_like(score_fake))
                adv_loss_mulit_band /= len(disc_fake_subband)

                # Frequency Discriminator
                adv_loss_freq = criterion(disc_fake_freq, torch.ones_like(disc_fake_freq))

                adv_loss = (adv_loss_multi_scale + adv_loss_mulit_band + adv_loss_freq)/3.0
            
                loss += opt.lambda_adv * adv_loss
                adv_loss = adv_loss.item() 

            loss.backward()
            optim.step()

            # discriminator
            if epoch > opt.discriminator_train_start_epoch:
                _, _, fake_audio = model(inp_freq)

                fake_audio = fake_audio.detach()
                optimizer_d.zero_grad()

                disc_fake, disc_subband_fake, disc_fake_freq = model_dsc(fake_audio)
                disc_real, disc_subband_real, disc_real_freq = model_dsc(gt_time.unsqueeze(1))
                
                # Time Domain multiscale Discriminator
                loss_d_real, loss_d_fake = 0.0, 0.0
                for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                    loss_d_real += criterion(score_real, torch.ones_like(score_real))
                    loss_d_fake += criterion(score_fake, torch.zeros_like(score_fake))
                loss_d_real = loss_d_real / len(disc_real) # len(disc_real) = 3
                loss_d_fake = loss_d_fake / len(disc_fake) # len(disc_fake) = 3

                # Time Domain subband Discriminator
                loss_d_subband_real, loss_d_subband_fake = 0.0, 0.0
                for (_, score_fake), (_, score_real) in zip(disc_subband_fake, disc_subband_real):
                    loss_d_subband_real += criterion(score_real, torch.ones_like(score_real))
                    loss_d_subband_fake += criterion(score_fake, torch.zeros_like(score_fake))
                loss_d_subband_real = loss_d_subband_real / len(disc_subband_real) # len(disc_real) = 3
                loss_d_subband_fake = loss_d_subband_fake / len(disc_subband_fake) # len(disc_fake) = 3

                # Frequency Discriminator
                loss_d_fake_freq = criterion(disc_fake_freq, torch.zeros_like(disc_fake_freq))
                loss_d_real_freq = criterion(disc_real_freq, torch.ones_like(disc_real_freq))

                loss_d = loss_d_real + loss_d_fake + loss_d_subband_real + loss_d_subband_fake + loss_d_fake_freq + loss_d_real_freq
                loss_d = loss_d / 6.0

                loss_d.backward()
                optim_d.step()
                loss_d = loss_d.item()

            ## log losses
            losses_std_avg += loss.item()
            losses_tb_avg += loss.item()
            loss_mel_tb_avg += loss_mel.item()
            loss_spec_tb_avg += loss_spec.item()
            loss_time_tb_avg += loss_time.item()
            loss_adv_tb_avg += adv_loss
            loss_dsc_tb_avg += loss_d

            std_ctr += 1
            tb_ctr += 1

            # STDOUT logging
            if steps % config.stdout_interval == 0:
                losses_std_avg /= std_ctr
                print(f"Steps: {steps}, Loss: {loss.item():4.3f}, Loss(Avg): {losses_std_avg:4.3f}, s/b: {time.time() - start_b:4.3f}")
                losses_std_avg = 0.0
                std_ctr = 0

            # checkpointing
            if steps % config.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = f"{config.checkpoint_path}/voc_{steps:08d}.pth"
                save_checkpoint(checkpoint_path, {
                    'model': model.state_dict(),
                    'steps': steps,
                    'epoch': epoch,
                    'optim': optim,
                    })
            
            # tensorboard summary logging
            if steps % config.summary_interval == 0:
                losses_tb_avg /= tb_ctr
                sw.add_scalar("training/loss_avg", losses_tb_avg, steps)
                sw.add_scalar("training/loss", loss.item(), steps)
                sw.add_scalar("training/loss_mel_avg", loss_mel_tb_avg, steps)
                sw.add_scalar("training/loss_spec_avg", loss_spec_tb_avg, steps)
                sw.add_scalar("training/loss_time_avg", loss_time_tb_avg, steps)
                sw.add_scalar("training/loss_generator", loss_adv_tb_avg, steps)
                sw.add_scalar("training/loss_discriminator", loss_dsc_tb_avg, steps)
                losses_tb_avg = 0.0
                loss_mel_tb_avg, loss_spec_tb_avg, loss_time_tb_avg = 0.0, 0.0, 0.0
                loss_adv_tb_avg, loss_dsc_tb_avg = 0.0, 0.0
                tb_ctr = 0

            if steps % config.validation_interval == 0:
                model.eval()
                torch.cuda.empty_cache()

                # validation error
                tot_val_loss = 0.0
                tot_val_loss_mel, tot_val_loss_spec, tot_val_loss_time = 0.0, 0.0, 0.0
                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        # prepare input
                        inp_val_time = batch['wav'].to(device).float()
                        inp_val_mel = get_mel_pytorch(inp_val_time, Config.n_fft, Config.hop_length,
                                        Config.win_size, window, mel_basis)

                        # output
                        pred_val_time = model(inp_val_mel, cuda=True)
                        
                        pred_val_time = pred_val_time.squeeze(1)
                        if pred_val_time.size(-1) > inp_val_time.size(-1):
                            pred_val_time = pred_val_time[:, :inp_val_time.size(-1)]
                        
                        pred_val_mel = get_mel_pytorch(pred_val_time, Config.n_fft, Config.hop_length,
                                        Config.win_size, window, mel_basis)

                        # loss calculation
                        loss_mel_val = mel_recon_loss(pred_val_mel, inp_val_mel)
                        loss_spec_val = spec_loss_fun(pred_val_time, inp_val_time)
                        loss_time_val = time_loss_fun(pred_val_time, inp_val_time)
                        
                        loss_val = config.lam_mel*loss_mel_val + loss_spec_val + loss_time_val

                        tot_val_loss_mel += loss_mel_val.item()
                        tot_val_loss_spec += loss_spec_val.item()
                        tot_val_loss_time += loss_time_val.item()
                        tot_val_loss += loss_val.item()
                
                tot_val_loss /= len(list(val_loader))
                tot_val_loss_mel /= len(list(val_loader))
                tot_val_loss_spec /= len(list(val_loader))
                tot_val_loss_time /= len(list(val_loader))

                print(f"Steps: {steps}, Val Loss: {tot_val_loss:4.3f}")

                sw.add_scalar("val/loss", tot_val_loss, steps)
                sw.add_scalar("val/loss_mel", tot_val_loss_mel, steps)
                sw.add_scalar("val/loss_spec", tot_val_loss_spec, steps)
                sw.add_scalar("val/loss_time", tot_val_loss_time, steps)

                # generate samples - check the generated samples is same for all runs
                for j, qidx in enumerate(qual_sample_idcs):
                    batch = val_loader.dataset[qidx]
                    
                    inp_time = torch.from_numpy(batch['wav']).to(device).float()
                    inp_time = inp_time.unsqueeze(0)

                    # convert the input time signal to melspectrogram
                    inp_mel = get_mel_pytorch(inp_time, Config.n_fft, Config.hop_length,
                                        Config.win_size, window, mel_basis) 

                    out_time = model(inp_mel, cuda=True)
                    
                    sw.add_audio(f"val/sample_{j}", out_time, steps, config.sample_rate)
                    sw.add_audio(f"val/sample_{j}_gt", inp_time, steps, config.sample_rate)

                model.train()
            
            steps += 1
            if steps >= config.training_steps:
                break
            torch.cuda.empty_cache()
    
    print(f"Time taken for epoch {epoch+1} is {int(time.time() - start)} sec\n")
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="output/vocoder_init") 
    parser.add_argument("--config", default="")
    parser.add_argument("--training_epochs", default=2000, type=int)
    parser.add_argument("--training_steps", default=400000, type=int)
    # parser.add_argument("--stdout_interval", default=20, type=int)
    # parser.add_argument("--checkpoint_interval", default=10000, type=int)
    # parser.add_argument('--summary_interval', default=100, type=int)
    # parser.add_argument('--validation_interval', default=2000, type=int)
    parser.add_argument("--use_lmdb", action="store_true")
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=10, type=int)
    parser.add_argument('--summary_interval', default=10, type=int)
    parser.add_argument('--validation_interval', default=20, type=int)

    a = parser.parse_args()
    h = load_config(a.config)
    build_env(a.config, "config.json", a.checkpoint_path)
    config = argparse.Namespace(**vars(a), **h)

    train(config)

if __name__ == "__main__":
    main()
