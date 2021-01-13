import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter
from tqdm import trange
from utils import generate_imgs, infiniteloop, set_seed

from model import Generator32
from model import Discriminator32

from evaluate.score.score import get_inception_and_fid_score

FLAGS = flags.FLAGS
# model & training
flags.DEFINE_integer('total_steps', 50000, "total number of training steps")
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.5, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 1, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 100, "latent space dimension")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 1000, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_bool('record', True, "record inception score and FID score")
flags.DEFINE_string('fid_cache', '../evaluate/pre_cal_stats/fid_stats_cifar10_train.npz', 'FID cache')
flags.DEFINE_string('logdir', './logs/CIFAR10', 'logging folder')

device = torch.device('cuda:0')

class BCEWithLogits(torch.nn.BCEWithLogitsLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            loss = super().forward(pred_real, torch.ones_like(pred_real))
            return loss

def train():
    dataset = datasets.CIFAR10(
            '~/datasets/data_cifar10', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4,
        drop_last=True)
    
    net_G = Generator32(FLAGS.z_dim).to(device)
    net_D = Discriminator32().to(device)
    loss_fn = BCEWithLogits()

    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)
    sched_G = optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1 - step / FLAGS.total_steps)
    sched_D = optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1 - step / FLAGS.total_steps)

    if not os.path.exists(os.path.join(FLAGS.logdir, "sample")):
        os.makedirs(os.path.join(FLAGS.logdir, "sample"))
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    with open(os.path.join(FLAGS.logdir, "FlagFile.txt"), "w") as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text("FlagFile", FLAGS.flags_into_string().replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2 # de-normalization
    writer.add_image('real_sample', grid)

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train the discriminator
            for _ in range(FLAGS.n_dis):
                with torch.no_grad():
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    fake = net_G(z).detach()
                real = next(looper).to(device)
                net_D_real = net_D(real)
                net_D_fake = net_D(fake)
                loss = loss_fn(net_D_real, net_D_fake)

                optim_D.zero_grad()
                loss.backward()
                optim_D.step()

                pbar.set_postfix(loss='%.4f' % loss)
            writer.add_scalar('loss', loss, step)
            # train the generator

            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device) # why batchsize * 2 ??
            loss = loss_fn(net_D(net_G(z)))

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            sched_D.step()
            sched_G.step()
            pbar.update(1)

            if step == 1 or step % FLAGS.sample_step == 0:
                fake = net_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))
            
            if step == 1 or step % FLAGS.eval_step == 0:
                torch.save({
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                }, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.record:
                    imgs = generate_imgs(
                        net_G, device, FLAGS.z_dim, 50000, FLAGS.batch_size)
                    is_score, fid_score = get_inception_and_fid_score(
                        imgs, device, FLAGS.fid_cache, verbose=True)
                    pbar.write(
                        "%s/%s Inception Score: %.3f(%.5f), "
                        "FID Score: %6.3f" % (
                            step, FLAGS.total_steps, is_score[0], is_score[1],
                            fid_score))
                    writer.add_scalar('inception_score', is_score[0], step)
                    writer.add_scalar('inception_score_std', is_score[1], step)
                    writer.add_scalar('fid_score', fid_score, step)
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    train()

if __name__ == '__main__':
    app.run(main)


