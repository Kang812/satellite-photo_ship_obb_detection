import cv2

def slide_window_prediction(img, stride, overlap):
    height, width = img.shape[:2]
    predictions = []
    scores = []
    for w in range(0, width + 1, stride - overlap):
        for h in range(0, height + 1, stride - overlap):
            start_w = w
            end_w = w + stride
            
            if end_w > width:
                end_w = width
                start_w = end_w - stride

            start_h = h
            end_h = h + stride
            
            if end_h > height:
                end_h = height
                start_h = end_h - stride

            crop_image  = img[start_h:end_h, start_w:end_w, :]
            cv2.imwrite(f"/workspace/oriented_dota/oriented_object_detection/sample_image/croping_image/{start_w}_{end_w}_{start_h}_{end_h}.jpg", crop_image)

if __name__ == '__main__':
    image_path = '/workspace/oriented_dota/oriented_object_detection/sample_image/competition_sample.png'
    
    img = cv2.imread(image_path)
    stride = 256
    overlap = 128

    slide_window_prediction(img, stride, overlap)
