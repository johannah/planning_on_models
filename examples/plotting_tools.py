
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_playout_scatters(true_env, base_path,  fname,
                         model_type,
                         seed, reward,rewards,actions,decision_time_estimates,
                         observed_frames, playout_frames,
                         model_road_maps,  rollout_length,
                         plot_error=True, gap=3, min_agents_alive=4, start_index=0,
                         do_plot_playouts=False, history_size=4):
    plt.ioff()
    pfpath = os.path.join(base_path,model_type,'E_%s'%(fname))

    if do_plot_playouts:
        if not os.path.exists(pfpath):
            os.makedirs(pfpath)

    fast_path = os.path.join(base_path,model_type,'T_%s'%(fname))
    if not os.path.exists(fast_path):
        os.makedirs(fast_path)

    start_state_index = playout_frames[0]['state_index']
    last_state_index = playout_frames[-1]['state_index']
    total_steps = start_state_index+len(playout_frames)


    fast_path = os.path.abspath(fast_path)
    fast_gif_path = os.path.join(fast_path, 'a_fast_seed_{}.gif'.format(seed))
    cmd = 'convert -delay 1/30 %s/*.png %s\n'%(fast_path, fast_gif_path)
    fast_sh_path = os.path.join(fast_path, 'run_fast_seed_{}.sh'.format(seed))
    of = open(fast_sh_path, 'w')
    of.write(cmd)
    of.close()

    game_num = 0
    game_step = 0
    total_num_games = len(actions)
    total_game_steps = len(actions[game_num])
    ts = 0
    try:
        for game_num in range(total_num_games):
            total_game_steps = len(actions[game_num])
            for game_step in range(total_game_steps):

                print('game state', game_num, game_step)
                state_index = ts
                true_obs = observed_frames[ts]
                print("plotting true frame {}/{} state_index {}/{} game {}/{}".format(ts,total_steps,state_index, last_state_index, game_num, total_num_games))
                true_state = prepare_img(true_obs)
                true_frame = true_env.get_state_plot(true_state)
                fast_fname = 'fast_seed_%06d_step_%05d.png'%(seed, state_index)
                ft,axt=plt.subplots(1,2, figsize=(7,3))
                axt[0].imshow(true_frame, vmin=0, vmax=255 )
                axt[0].set_title("game num %03d/%03d game step %03d/%03d "%(game_num, total_num_games, game_step, total_game_steps))

                axt[1].imshow(decision_time_estimates[ts], vmin=0, vmax=255 )
                axt[1].set_title(" model - real step:%05d/%05d "%(ts, total_steps))
                ft.tight_layout()
                plt.savefig(os.path.join(fast_path,fast_fname))
                plt.close()

                embed()
                ts +=1

    except Exception, e:
        print(e, 'plot')
        embed()

    #print('writing gif')
    #try:
    #    subprocess.call(['sh', fast_sh_path])
    #except Exception, e:
    #    print(e); embed()
    print("FINISHED WRITING TO", fast_sh_path)


    if do_plot_playouts:

        for ts, step_frame in enumerate(playout_frames):
            state_index = step_frame['state_index']
            print("plotting true frame {}/{} state_index {}/{}".format(ts,total_steps,state_index, last_state_index))
            true_obs = observed_frames[state_index]
            true_state = prepare_img(true_obs)
            true_frame = true_env.get_state_plot(true_state)
            # playtouts is size episode_length, y, x
            c = 0
            playout_agent_states = step_frame['playout_agent_states']
            playout_agent_locs_y = step_frame['playout_agent_locs_y']
            playout_agent_locs_x = step_frame['playout_agent_locs_x']
            playout_model_states = step_frame['playout_model_states']
            num_playout_steps = playout_model_states.shape[0]
            plot_inds = range(0,num_playout_steps,gap)
            if (num_playout_steps-1) not in plot_inds:
                plot_inds.append(num_playout_steps-1)
            for playout_ind in plot_inds:
                playout_state_index = min(state_index+playout_ind, ref_frames_prep.shape[0]-1)
                print("plotting playout state_index {}/{} - {} step {}/{}".format(
                                                state_index, total_steps, playout_state_index, playout_ind, num_playout_steps))
                ref_ind = min(((start_index+(ts*args.frame_skip)) + (playout_ind*args.frame_skip)), ref_frames_prep.shape[0]-1)
                print("step",ts,'ref', ref_ind, 'play',playout_ind)
                true_playout_frame = ref_frames_prep[ref_ind]
                est_playout_frame = playout_model_states[playout_ind]
                _, rollout_model_error  = get_false_neg_counts(deepcopy(true_playout_frame), deepcopy(est_playout_frame))
                fname = 'seed_%06d_tstep_%04d_pstep_%04d_ps_%04d.png'%(seed, state_index, playout_state_index, playout_ind)
                f,ax=plt.subplots(1,4, figsize=(10,3.1))
                ax[0].imshow(true_frame, vmin=0, vmax=255 )
                ax[0].set_title("decision t: {}/{}".format(state_index,last_state_index))
                ax[1].imshow(true_playout_frame, vmin=0, vmax=255 )
                ax[1].set_title("oracle rollout %04d step:%03d %03d/%03d"%(state_index, playout_state_index, playout_ind, num_playout_steps))
                ax[2].imshow(est_playout_frame,  vmin=0, vmax=255 )
                ax[2].set_title("model rollout")
                ax[3].imshow(rollout_model_error, cmap='Set1')
                ax[3].set_title("error in model")
                if playout_ind in playout_agent_locs_y.keys():
                    agent_x = playout_agent_locs_x[playout_ind]
                    agent_y = playout_agent_locs_y[playout_ind]
                    ax[1].scatter(agent_x, agent_y, alpha=0.5, s=4, c='y')
                    ax[2].scatter(agent_x, agent_y, alpha=0.5, s=4, c='y')

                f.tight_layout()
                plt.savefig(os.path.join(pfpath,fname))
                plt.close()
        print("making gif")
        gif_path = 'a_seed_{}.gif'.format(seed)
        search = os.path.join(pfpath, 'seed_*.png')
        cmd = 'convert -delay 1/100000 *.png %s \n'%( gif_path)
        sh_path = os.path.join(pfpath, 'run_seed_{}.sh'.format(seed))
        sof = open(sh_path, 'w')
        sof.write(cmd)
        sof.close()
        print("FINISHED slow WRITING TO", os.path.split(pfpath)[0])


if __name__ == "__main__":
    import argparse
    vq_name = 'nfreeway_vqvae4layer_nl_k512_z64e00250_good.gpkl'
    #pcnn_name = 'mrpcnn_id512_d256_l10_nc4_cs1024___k512_z64e00034_good.gpkl'
    pcnn_name = 'erpcnn_id512_d256_l10_nc4_cs1024___k512_z64e00040_good.gpkl'

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-bs', '--buffer_size', type=int, default=5, help='buffer size around q value')
    parser.add_argument('-y', '--ysize', type=int, default=48, help='pixel size of game in y direction')
    parser.add_argument('-x', '--xsize', type=int, default=48, help='pixel size of game in x direction')
    parser.add_argument('-g', '--max_goal_distance', type=int, default=1000, help='limit goal distance to within this many pixels of the agent')
    parser.add_argument('-l', '--level', type=int, default=6, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-p', '--num_playouts', type=int, default=200, help='number of playouts for each step')
    parser.add_argument('-r', '--rollout_steps', type=int, default=10, help='limit number of steps taken be random rollout')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print debug info')
    parser.add_argument('-t', '--model_type', type=str, default='vqvae_pcnn_model')
    parser.add_argument('-msf', '--env_model_type', type=str, default='equiv_model_step')
    parser.add_argument('-sams', '--num_samples', type=int , default=5)
    parser.add_argument('-gs', '--goal_speed', type=float , default=0.5)
    parser.add_argument('-neo', '--neo_goal_prior', type=float , default=0.01)
    parser.add_argument('-sm', '--smoothing', type=float , default=0.5)
    parser.add_argument('-sl', '--step_limit', type=float , default=18000)
    parser.add_argument('-as', '--agent_max_speed', type=float , default=1.0)
    parser.add_argument('-fre', '--full_rollouts_every', type=float , default=10)
    parser.add_argument('-fs', '--frame_skip', type=float , default=4)

    parser.add_argument('--save_pkl', action='store_false', default=True)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('-gap', '--plot_playout_gap', type=int, default=5, help='gap between plot playouts for each step')
    parser.add_argument('-f', '--prior_fn', type=str, default='goal', help='options are goal or equal')

    args = parser.parse_args()
    if args.plot_playouts:
        args.save_plots = True
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    global buf
    buf = np.ones((args.buffer_size * 2, args.buffer_size * 2))
    args.full_rollouts_every = min(args.full_rollouts_every, args.rollout_steps)
    goal_dis = args.max_goal_distance
    if args.prior_fn == 'goal':
        prior = goal_node_probs_fn
    else:
        prior = equal_node_probs_fn

    use_cuda = args.cuda
    seed = args.seed
    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    ref_file = 'reference_freeway.npz'
    ref_env = gym.make('FreewayNoFrameskip-v4')
    if os.path.exists(ref_file):
        print("loading reference frames")
        rf = np.load(open(ref_file, 'r'))
        ref_frames = rf['ref_frames']
        ref_frames_prep = rf['ref_frames_prep']
    else:
        print("creating reference frames")
        lo = ref_env.reset()
        _, ref_tobs = prepare_img(lo)
        ref_frames = [lo]
        ref_frames_prep = [ref_tobs]
        f = False
        while not f:
            o, r, f, _ = ref_env.step(0)
            om = np.maximum(o, lo)
            _,to = prepare_img(om)
            ref_frames.append(om)
            ref_frames_prep.append(to)
            lo = o
        np.savez(open(ref_file, 'w'), ref_frames=ref_frames, ref_frames_prep=ref_frames_prep)
    #ref_env = gym.make('FreewayNoFrameskip-v4')
    #lo = ref_env.reset()
    #aref_frames = [lo]
    #renv, observ, lo, reward, done, ref_state_index = frame_skip_step(ref_env, lo, 0, 0, False, 500, 0)
    #embed()

    DIM = 256
    history_size = 4
    cond_size = history_size*DIM
    upcnn_name  = 'na'
    uvq_name = 'na'
    if args.model_type == 'vqvae_pcnn_model':
        dsize = 80
        nr_logistic_mix = 10
        probs_size = (2*nr_logistic_mix)+nr_logistic_mix
        num_z = 64
        nr_logistic_mix = 10
        num_clusters = 512
        N_LAYERS = 10 # layers in pixelcnn

        upcnn_name  = pcnn_name.split('e00')[1].replace('.pkl', '')
        uvq_name  = vq_name.split('e00')[1].replace('.pkl', '')
        default_pcnn_model_loadpath = os.path.join(config.model_savedir, pcnn_name)
        default_vqvae_model_loadpath = os.path.join(config.model_savedir, vq_name)
        if os.path.exists(default_vqvae_model_loadpath):
            vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix,num_clusters=num_clusters, encoder_output_size=num_z).to(DEVICE)
            vqvae_model_dict = torch.load(default_vqvae_model_loadpath, map_location=lambda storage, loc: storage)
            vmodel.load_state_dict(vqvae_model_dict['state_dict'])
            epoch = vqvae_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch,
                                                   default_vqvae_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(default_vqvae_model_loadpath))
            sys.exit()

        if os.path.exists(default_pcnn_model_loadpath):
            pcnn_model = GatedPixelCNN(num_clusters, DIM, N_LAYERS,
                                        history_size, spatial_cond_size=cond_size).to(DEVICE)
            pcnn_model_dict = torch.load(default_pcnn_model_loadpath, map_location=lambda storage, loc: storage)
            pcnn_model.load_state_dict(pcnn_model_dict['state_dict'])
            epoch = pcnn_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch,
                                                   default_pcnn_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(default_pcnn_model_loadpath))
            sys.exit()


    action_space = range(ref_env.action_space.n)
    model_env = FreewayEnv(action_space, model_step_fn=args.env_model_type, frame_skip=args.frame_skip)
    for start_index in range(30):
        # load relevent file
        fname = 'acbafay_seed_%03d_si_%02d_sl_%09d_p%03d_r%03d_pr_%s_mod_%s_vq_%s_pcnn_%s_sam_%s_neo_%01.02f_bs_%02d_fre_%03d_smooth_%.02f_%s.pkl' %(
                                    seed,
                                    start_index,
                                    args.step_limit,
                                    args.num_playouts,
                                    args.rollout_steps,
                                    args.prior_fn,
                                    args.model_type,
                                    uvq_name,
                                    upcnn_name,
                                    args.num_samples,
                                    args.neo_goal_prior,
                                    args.buffer_size,
                                    args.full_rollouts_every,
                                    args.smoothing,
                                    args.env_model_type)

        if not os.path.exists(config.results_savedir):
            os.makedirs(config.results_savedir)
        fpath = os.path.join(config.results_savedir, fname)
        if not os.path.exists(fpath):
            all_results = {'args':args}
            print("STARTING EPISODE start_index %s" %(start_index))
            print(args.save_pkl)
            st = time.time()
            r = run_trace(fname, seed=seed,
                          n_playouts=args.num_playouts,
                          max_rollout_length=args.rollout_steps,
                          prob_fn=prior, estimator=args.model_type,
                          history_size=history_size, start_index=start_index, do_render=args.render)

            et = time.time()
            r['full_end_time'] = et
            r['full_start_time'] = st
            r['seed'] = seed
            all_results[start_index] = r
            if args.save_pkl:
                ffile = open(fpath, 'w')
                pickle.dump(all_results,ffile)
                print("saved start_index %s"%start_index)
                ffile.close()
    embed()
    print("FINISHED")


plt.clf()
plt.close()
if args.save_plots:
    plot_playout_scatters(model_env, os.path.join(config.results_savedir, 'trials'), fname.replace('.pkl',''),
                      str(estimator), seed, reward, rewards,
                      results['actions'], decision_time_estimates,
                      observed_frames=observ_frames,
                      playout_frames=playout_frames,
                      model_road_maps=pmcts.road_map_ests,
                      rollout_length=pmcts.rollout_length,
                      plot_error=args.do_plot_error,
                      gap=args.plot_playout_gap,
                      min_agents_alive=4,start_index=start_index,
                      do_plot_playouts=args.plot_playouts,
                      history_size=history_size)

if args.plot_playouts:
    playout_frames.append({'state_index':state_index,
                           'playout_agent_states':deepcopy(pmcts.playout_agents),
                           'playout_agent_locs_y':deepcopy(pmcts.playout_agent_locs_y),
                           'playout_agent_locs_x':deepcopy(pmcts.playout_agent_locs_x),
                           'playout_model_states':deepcopy(pmcts.playout_road_maps),
                           })



