from torchvision.transforms.functional import crop


def crop_transform(
    image, camera_view=0, image_size=480
):  # This is significant to the setup
    if camera_view == 0:
        return crop(image, 0, 80, image_size, image_size)
    if camera_view == 1:
        return crop(image, 0, 80, image_size, image_size)
    # elif camera_view == 1: # NOTE: For now we don't have any other views
    #     return crop(image, 0,90, image_size,image_size)
    elif camera_view == 2:
        return crop(image, 0, 40, image_size, image_size)

    print(f"crop_transform returning None for camera_view={camera_view}")
