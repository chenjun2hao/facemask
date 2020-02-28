import facemask

if __name__ == '__main__':
    path = './model_best.pth'
    image_path = './test.png'
    image_path = './test_00000243.jpg'
    model = facemask.FaceMaskDetector(path)
    # res = model(image_path)
    res = model.detect_image_show(image_path)
    print(res)