
def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
#
def get_false_neg_counts(true_road_map, pred_road_map):
    road_true_road_map = deepcopy(true_road_map)
    road_pred_road_map = deepcopy(pred_road_map)
    road_true_road_map[road_true_road_map>0] = 1
    road_pred_road_map[road_pred_road_map>0] = 1
    # false_neg predict free where there was car # bad
    false_neg = (road_true_road_map*np.abs(road_true_road_map-road_pred_road_map))
    false_neg[false_neg>0] = 1
    false_neg_count = false_neg.sum()
    false_pos = road_pred_road_map*np.abs(road_true_road_map-road_pred_road_map)
    false_neg = road_true_road_map*np.abs(road_true_road_map-road_pred_road_map)
    error = np.ones_like(road_true_road_map)*254
    error[false_pos>0] = 30
    error[false_neg>0] = 1
    return false_neg_count, error


