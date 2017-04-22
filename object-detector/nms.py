def IOU(box1,box2):
    #one box is represented by a tuple:[Xmin,Ymin,Xmax,Ymax]
    x1_tl = box1[0]
    x2_tl = box2[0]
    x1_br = box1[2]
    x2_br = box2[2]
    y1_tl = box1[1]
    y2_tl = box2[1]
    y1_br = box1[3]
    y2_br = box2[3]
    w1 = x1_br - x1_tl
    h1 = y1_br - y1_tl
    w2 = x2_br - x2_tl
    h2 = y2_br - y2_tl
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = w1 * h1
    area_2 = w2 * h2
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(predictions, threshold=.5):
    '''
    This function performs Non-Maxima Suppression.
    `predictions` consists of a list of predictions.
    Each prediction is in the format ->
    [x-top-left, y-top-left, confidence-of-predictions, width-of-prediction, height-of-prediction]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of predictions.
    '''
    if len(predictions) == 0:
        return []
    # Sort the predictions based on confidence score
    predictions = sorted(predictions, key=lambda predictions: predictions[4],
            reverse=True)
    # Store suppressed predictions
    suppressed=[]
    # Unique predictions will be appended to this list
    new_predictions=[]
    for i in range(len(predictions)):
        for j in range(i+1,len(predictions)) :
            if j not in suppressed and IOU(predictions[i], predictions[j]) > threshold:
                suppressed.append(j)
    for i in range(len(predictions)) :
        if i not in suppressed:
            new_predictions.append(predictions[i])
    return new_predictions

if __name__ == "__main__":
    # Example of how to use the NMS Module
    predictions = [[31, 31, 41, 41, .9], [31, 31, 41, 41, .12], [100, 34, 110, 44, .8]]
    print("predictions before NMS = {}".format(predictions))
    print("predictions after NMS = {}".format(nms(predictions)))
