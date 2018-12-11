
def get_zero_model(state_index, est_inds, cond_states):
    rollout_length = len(est_inds)
    print("starting none %s predictions" %rollout_length)
    # normalize data before putting into vqvae
    return np.zeros_like(ref_frames_prep[est_inds])

def get_none_model(state_index, est_inds, cond_states):
    rollout_length = len(est_inds)
    print("starting none %s predictions for state index %s " %(len(est_inds), state_index))
    # normalize data before putting into vqvae

    scaled_est_inds = np.array([(start_index+(e*args.frame_skip)) for e in est_inds])
    max_frame_ind = ref_frames_prep.shape[0]-1
    scaled_est_inds[scaled_est_inds>max_frame_ind] = max_frame_ind

    #print('forward for state index', state_index)
    #print(est_inds)
    #print(scaled_est_inds)
    return ref_frames_prep[scaled_est_inds]


