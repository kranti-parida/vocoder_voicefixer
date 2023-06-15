import torch
import os
import argparse
import json
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from dataset import wav_mel_dataset, get_dataset
from model.generator import Generator
from model.util import load_try


from config import Config
from utils.utils import load_checkpoint, scan_checkpoint, save_checkpoint, build_env, AttrDict

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
        check_cuda_availability(cuda=cuda)
        self.model = try_tensor_cuda(self.model, cuda=cuda)
        mel = try_tensor_cuda(mel, cuda=cuda)
        self.weight_torch = self.weight_torch.type_as(mel)
        # mel = mel / self.weight_torch
        mel = tr_normalize(tr_amp_to_db(torch.abs(mel)) - 20.0)
        mel = tr_pre(mel[:, 0, ...])
        wav_re = self.model(mel)
        return wav_re

def load_config(filepath):
    with open(filepath) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h

def train(config):
    torch.cuda.manual_seed(config.seed)
    device = torch.device("cuda")
    
    # load model
    model = Vocoder(config.sample_rate)

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
    val_dataset = wav_mel_dataset(config, val_paths)

    train_loader = DataLoader(train_dataset, num_workers=0, shuffle=True, 
                                batch_size=config.batch_size, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=0, shuffle=False, 
                                batch_size=config.batch_size, pin_memory=True, drop_last=False)

    sw = SummaryWriter(os.path.join(config.checkpoint_path, 'logs'))

    model.train()

    ## loss function
    losses_std_avg = 0.0
    losses_tb_avg = 0.0
    std_ctr = 0
    tb_ctr = 0

    for epoch in range(max(0, last_epoch), config.train_epochs):
        start = time.time()
        print(f"Epoch: {epoch}")

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            inp_time = batch['wav'].to(device)
            
            import pdb; pdb.set_trace()
            ## convert the input time signal to melspectrogram
            # inp_mel 

            out_time = model(inp_mel, cuda=True)

            optim.zero_grad()

            loss = loss_fn(out_time, inp_time)

            losses_std_avg += loss.item()
            losses_tb_avg += loss.item()
            std_ctr += 1
            tb_ctr += 1

            loss.backward()
            optim.step()

            # STDOUT logging
            if steps % config.stdout_interval == 0:
                losses_std_avg /= std_ctr
                print(f"Steps: {steps}, Loss: {loss.item():4.3f}, Loss(Avg): {losses_std_avg:4.3f}, s/b: {time.time() - start_b:4.3f}")
                losses_std_avg = 0.0
                std_ctr = 0

            # checkpointing
            if steps % config.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = f"{config.checkpoint_path}/voc_{steps:08d}"
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
                losses_tb_avg = 0.0
                tb_ctr = 0

            if steps % config.validation_interval == 0:
                model.eval()
                torch.cuda.empty_cache()

                # validation error
                tot_val_loss = 0.0
                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        # prepare input
                        inp_time = batch['wav'].to(device)

                        # convert the input time signal to melspectrogram
                        # inp_mel 

                        out_time = model(inp_mel, cuda=True)
                        loss = loss_fn(out_time, inp_time)
                        tot_val_loss += loss.item()
                
                tot_val_loss /= len(list(val_loader))
                sw.add_scalar("val/loss", tot_val_loss, steps)

                # generate samples - check the generated samples is same for all runs
                
                for j, qidx in enumerate(qual_sample_idcs):
                    # prepare input
                    inp_time = batch['wav'].to(device)

                    # convert the input time signal to melspectrogram
                    # inp_mel 

                    out_time = model(inp_mel, cuda=True)
                    
                    sw.add_audio(f"val/sample_{j}", out_time, steps, config.sample_rate)
                    sw.add_audio(f"val/sample_{j}_gt", inp_time, steps, config.sample_rate)

                model.train()
            
            steps += 1
            if steps >= a.training_steps:
                break
    
    print(f"Time taken for epoch {epoch+1} is {int(time.time() - start)} sec\n")
        


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", default="output/vocoder_init") 
    parser.add_argument("--config", default="")
    parser.add_argument("--training_epochs", default=2000, type=int)
    parser.add_argument("--training_steps", default=400000, type=int)
    parser.add_argument("--stdout_interval", default=20, type=int)
    parser.add_argument("--checkpoint_interval", default=10000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=2000, type=int)

    a = parser.parse_args()
    h = load_config(a.config)
    build_env(a.config, "config.json", a.checkpoint_path)
    config = argparse.Namespace(**vars(a), **h)

    train(config)

if __name__ == "__main__":
    main()