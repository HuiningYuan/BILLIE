import argparse
import torch
import numpy as np
import torch.optim as optim
import model
import utils
import build_library
import scipy.io
import time
import os
import json

# Parameters
parser = argparse.ArgumentParser(description='BILLIE demo')
parser.add_argument('--data', type=str, default='Navier-Stokes', help='Burgers or Navier-Stokes')
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--RNN_hidden_size', type=int, default=1024)
parser.add_argument('--RNN_max_steps', type=int, default=20)
parser.add_argument('--log', type=str, default='./log.txt')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--l2_s1_weight', nargs='+', type=float, default=[0, 0],
                    help='Weight of l2 loss on s1 set for reward calculating, typically set to 0. Such weight can be set separatly for each component.')
parser.add_argument('--l2_s2_weight', nargs='+', type=float, default=[1e5, 1e5],
                    help='Weight of l2 loss on s2 set for reward calculating. Such weight can be set separatly for each component.')
parser.add_argument('--l0_weight', nargs='+', type=float, default=[1, 1],
                    help='Weight of sparse regularization for reward calculating. Such weight can be set separatly for each component.')
parser.add_argument('--entropy_weight', type=float, default=1,
                    help='Weight of entropy in the final loss function to encourage agent exploration')
parser.add_argument('--baseline_init_epochs', type=int, default=100)
args = parser.parse_args()

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
    device = 'cuda'
else:
    device = 'cpu'

# Plant seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Build data
if args.data == 'Burgers':
    # Load data and build over-complete library
    noise_level = 0.05
    D = 3
    P = 3
    # Load s1 set
    data_s1 = scipy.io.loadmat('./data/burgers_s1.mat')
    u_s1 = np.real(data_s1['u'])
    u_s1 = u_s1.T
    x_s1 = np.real(data_s1['x'][0])
    t_s1 = np.real(data_s1['t'][0])
    dt_s1 = t_s1[1] - t_s1[0]
    dx_s1 = x_s1[2] - x_s1[1]
    Ut_s1, Q_s1, _ = utils.build_burgers_data(u_s1, dx_s1, dt_s1, D, P, noise_level)

    # Load s2 set
    data_s2 = scipy.io.loadmat('./data/burgers_s2.mat')
    u_s2 = np.real(data_s2['u'])
    u_s2 = u_s2.T
    x_s2 = np.real(data_s2['x'][0])
    t_s2 = np.real(data_s2['t'][0])
    dt_s2 = t_s2[1] - t_s2[0]
    dx_s2 = x_s2[2] - x_s2[1]
    Ut_s2, Q_s2, description = utils.build_burgers_data(u_s1, dx_s1, dt_s1, D, P, noise_level)
elif args.data == 'Navier-Stokes':
    # data_s1 = np.load('./data/data_2048_nof_Re1000_100steps_0.npy')
    # data_s1 = data_s1.transpose((1, 2, 0, 3))
    # U_s1 = data_s1[:, :, :, 1]
    # V_s1 = data_s1[:, :, :, 0]
    # P_s1 = data_s1[:, :, :, 2]
    # U_s1 = 0.5 * (U_s1 + np.roll(U_s1, -1, axis=0))
    # V_s1 = 0.5 * (V_s1 + np.roll(V_s1, 1, axis=1))
    # data_s2 = np.load('./data/data_2048_nof_Re1000_100steps_2.npy')
    # data_s2 = data_s2.transpose((1, 2, 0, 3))
    # U_s2 = data_s2[:, :, :, 1]
    # V_s2 = data_s2[:, :, :, 0]
    # P_s2 = data_s2[:, :, :, 2]
    # U_s2 = 0.5 * (U_s2 + np.roll(U_s2, -1, axis=0))
    # V_s2 = 0.5 * (V_s2 + np.roll(V_s2, 1, axis=1))
    # n = U_s1.shape[0]
    # m = U_s1.shape[1]
    # steps = U_s1.shape[2]
    # dt = 0.00005
    # dx = 2 * np.pi / n
    # dy = 2 * np.pi / m
    # num_xy = 100
    # num_t = 30
    # num_points = num_xy * num_t
    # boundary_x = 7
    # boundary_y = 7
    # boundary_t = 7
    # deg = 5  # degree of polynomial to use
    # points = {}
    # count = 0
    # spatial_points = np.sort(np.random.choice((n - 2 * boundary_x) * (m - 2 * boundary_y), num_xy, replace=False))
    # temporal_points = np.sort(np.random.choice(steps - 2 * boundary_t, num_t, replace=False))
    # for i in range(num_xy):
    #     x = spatial_points[i] // (n - 2 * boundary_x)
    #     y = spatial_points[i] % (n - 2 * boundary_x)
    #     for j in range(num_t):
    #         points[count] = [x + boundary_x, y + boundary_y, temporal_points[j] + boundary_t]
    #         count += 1
    # Ut_s1, Q_s1, _ = utils.build_ns_data(U_s1, V_s1, P_s1, points, num_points=num_points, dx=dx, dy=dy, dt=dt,
    #                                      boundary_x=boundary_x, boundary_y=boundary_y,
    #                                      boundary_t=boundary_t, deg=deg)
    # Ut_s2, Q_s2, description = utils.build_ns_data(U_s1, V_s1, P_s1, points, num_points=num_points,
    #                                                dx=dx, dy=dy, dt=dt,
    #                                                boundary_x=boundary_x, boundary_y=boundary_y,
    #                                                boundary_t=boundary_t, deg=deg)
    Ut_s1 = np.load('./data/NS_Ut_s1.npy')
    Q_s1 = np.load('./data/NS_Q_s1.npy')
    Ut_s2 = np.load('./data/NS_Ut_s2.npy')
    Q_s2 = np.load('./data/NS_Q_s2.npy')
    description = description_val = np.load('./data/NS_description.npy')

# RNN Parameters
num_of_components = len(Ut_s1)
embedding_size = args.RNN_hidden_size
hidden_size = args.RNN_hidden_size

# PDE Parameters
p = len(description)  # Number of candidates in the over-complete library
n_actions = p  # Action space

# Build model
model = model.BILLIE(n_actions=n_actions,
                     num_of_components=num_of_components,
                     embedding_size=embedding_size,
                     hidden_size=hidden_size,
                     device=device)

# Build log file
log_file = args.log
f = open(log_file, mode='w+')
f.writelines(('max steps: {:d}\n'.format(args.RNN_max_steps),
              'entropy: {:.2e}\n'.format(args.entropy_weight),
              'lr: {:.2e}\n'.format(args.lr)))

# Build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Initialize baseline
print('Initializing baseline...')
baseline = []
for v in range(num_of_components):
    print('Initializing baseline for component No. {:d}'.format(v))
    baseline_buffer = []
    for i in range(args.baseline_init_epochs):
        _, _, buffer_candidates = model.autoregression(args.RNN_max_steps,
                                                       component=v,
                                                       cuda=args.cuda)
        if len(buffer_candidates) > 1:
            reward, l2_loss_train, l2_loss_val, l0_loss = utils.get_reward_log(
                tgt_train=Ut_s1[v],
                library_train=Q_s1,
                candidates=buffer_candidates[
                           1:].cpu().numpy().tolist(),
                tgt_val=Ut_s2[v],
                library_val=Q_s2,
                a1=args.l2_s1_weight[v],
                a2=args.l2_s2_weight[v],
                a3=args.l0_weight[v])
            baseline_buffer.append(reward)
    baseline.append(sum(baseline_buffer) / len(baseline_buffer))
print('Baseline initialization complete!')

# Train
all_reward = []
top5_candidates = [[] for i in range(num_of_components)]
print('Starting training...')
t = time.time()
for i in range(args.epochs):
    print('Epoch No. {:d}'.format(i))
    f.writelines('Epoch No. {:d} \n'.format(i))
    # Use the agent in a auto-regressive way
    buffer_loss_all = []
    for v in range(num_of_components):
        # buffers
        buffer_reward = []
        buffer_loss = []
        for b in range(args.batch_size):
            # Build inputs
            buffer_log_probs, buffer_entropy, buffer_candidates = model.autoregression(args.RNN_max_steps,
                                                                                       component=v,
                                                                                       cuda=args.cuda)
            if len(buffer_candidates) > 1:
                reward, l2_loss_train, l2_loss_val, l0_loss = utils.get_reward_log(
                    tgt_train=Ut_s1[v],
                    library_train=Q_s1,
                    candidates=buffer_candidates[
                               1:].cpu().numpy().tolist(),
                    tgt_val=Ut_s2[v],
                    library_val=Q_s2,
                    a1=args.l2_s1_weight[v],
                    a2=args.l2_s2_weight[v],
                    a3=args.l0_weight[v])
            else:
                reward = 1.2 * baseline[v]

            # save reward
            all_reward.append(reward)
            buffer_reward.append(reward)
            tgt_function = (reward - baseline[v]) * sum(buffer_log_probs)
            entropy = sum(buffer_entropy).sum() / n_actions
            buffer_loss.append(-tgt_function + args.entropy_weight * entropy)
            print('Run {:02d} of variable No. {:d}'.format(b, v),
                  'l0: {:02d}'.format(l0_loss),
                  'l2_train: {:.8f}'.format(l2_loss_train),
                  'l2_val: {:.8f}'.format(l2_loss_val),
                  'Reward: {:.8f} '.format(reward),
                  'Baseline: {:.8f}'.format(baseline[v]),
                  'Advantage: {:.8f}'.format(reward - baseline[v]),
                  'time: {:.4f}'.format(time.time() - t))
            f.writelines(('Run {:d} of variable No. {:d}   '.format(b, v),
                          'l0: {:02d}   '.format(l0_loss),
                          'l2_train: {:.8f}   '.format(l2_loss_train),
                          'l2_val: {:.8f}   '.format(l2_loss_val),
                          'Reward: {:.8f}   '.format(reward),
                          'Baseline: {:.8f}   '.format(baseline[v]),
                          'Advantage: {:.8f}   '.format(reward - baseline[v]),
                          'time: {:.4f}\n'.format(time.time() - t)))
        buffer_loss_all.append(sum(buffer_loss))
        baseline[v] = 0.9 * baseline[v] + 0.1 * sum(buffer_reward) / len(buffer_reward)
        print('\n')

    print('Updating...\n')
    f.writelines(('Updating...\n\n'))
    t = time.time()
    optimizer.zero_grad()
    loss = sum(buffer_loss_all)
    loss.backward()
    optimizer.step()
    # Print result every 20 epochs
    if i % 20 == 19:
        # Lambda
        lam = [args.l0_weight[i] / (args.l2_s1_weight[i] + args.l2_s2_weight[i]) for i in range(num_of_components)]
        print('a1: ', json.dumps(args.l2_s1_weight), '\n',
              'a2: ', json.dumps(args.l2_s2_weight), '\n',
              'a3: ', json.dumps(args.l0_weight), '\n',
              'lambda: ', json.dumps(lam), '\n')
        f.writelines(('a1: ', json.dumps(args.l2_s1_weight), '\n',
                      'a2: ', json.dumps(args.l2_s2_weight), '\n',
                      'a3: ', json.dumps(args.l0_weight), '\n',
                      'lambda: ', json.dumps(lam), '\n\n'))
        # Test
        # Use the transformer in a auto-regressive way
        for v in range(num_of_components):
            # Build inputs
            buffer_log_probs, buffer_entropy, buffer_candidates = model.autoregression(args.RNN_max_steps,
                                                                                       component=v,
                                                                                       cuda=args.cuda)
            candidates = buffer_candidates[1:].cpu().numpy().tolist()
            final_candidates, weight = utils.get_weight(Ut_s1[v], Q_s1, candidates)

            flag = 0
            for i in range(len(top5_candidates[v])):
                if final_candidates == top5_candidates[v][i]:
                    flag = 1
                    break
            if flag == 0:
                top5_rewards = []
                top5_candidates[v].append(final_candidates)
                for i in range(len(top5_candidates[v])):
                    reward, _, _, _ = utils.get_reward_log(tgt_train=Ut_s1[v],
                                                           library_train=Q_s1,
                                                           candidates=top5_candidates[v][i],
                                                           tgt_val=Ut_s2[v],
                                                           library_val=Q_s2,
                                                           a1=args.l2_s1_weight[v],
                                                           a2=args.l2_s2_weight[v],
                                                           a3=args.l0_weight[v])
                    top5_rewards.append(reward)
                sort = [index for index, value in
                        sorted(list(enumerate(top5_rewards)), key=lambda x: x[1], reverse=True)]
                temp = [top5_candidates[v][i] for i in sort]
                top5_candidates[v] = temp

            # Print the PDE
            if len(final_candidates) > 0:
                print('\nPDE of variable No. {:d} is written as:'.format(v))
                f.writelines(('\n\nPDE of variable No. {:d} is written as:\nxt = '.format(v)))
                print('xt = ', end='')
                for i in range(len(final_candidates) - 1):
                    print('%f * %s + ' % (weight[i], description[final_candidates[i]]), end='')
                    f.writelines('%f * %s + ' % (weight[i], description[final_candidates[i]]))
                print('%f * %s\n' % (weight[-1], description[final_candidates[-1]]))
                f.writelines('%f * %s\n\n' % (weight[-1], description[final_candidates[-1]]))
            else:
                print('No PDE for variable No. {:d}\n'.format(v))
                f.writelines(('No PDE for variable No. {:d}\n\n'.format(v)))

            # Print top 5
            print('Current top 5 are:')
            f.writelines(('Current top 5 are:\n'))
            for i in range(min(len(top5_candidates[v]), 5)):
                candidates = top5_candidates[v][i]
                final_candidates, weight = utils.get_weight(Ut_s1[v], Q_s1, candidates)
                if len(final_candidates) > 0:
                    print('xt = ', end='')
                    f.writelines(('xt = '.format(v)))
                    for i in range(len(final_candidates) - 1):
                        print('%f * %s + ' % (weight[i], description[final_candidates[i]]), end='')
                        f.writelines('%f * %s + ' % (weight[i], description[final_candidates[i]]))
                    print('%f * %s' % (weight[-1], description[final_candidates[-1]]))
                    f.writelines('%f * %s\n' % (weight[-1], description[final_candidates[-1]]))
                else:
                    print('No PDE for variable No. {:d}'.format(v))
                    f.writelines(('No PDE for variable No. {:d}\n'.format(v)))
# Close log file
f.close()
