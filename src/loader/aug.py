import cv2
import random
import numpy as np
from io import BytesIO
from PIL import Image
from scipy.ndimage import gaussian_filter

class DataAug:
    
    def __init__(self, patch_size=32, thresh = 0.65, **kwargs):
        self.thresh = thresh
        self.patch_size = patch_size
        self.simple_choices = np.array(list(range(0, 4)))
        self.flip_simple_choices = list(range(0, 2))
        
        
    def shuffle_patches(self, image, patch_size=32):
        # Ensure the image dimensions are divisible by the patch size
        h, w, c = image.shape
        assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by the patch size."
        
        # Calculate the number of patches along each dimension
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        
        # Reshape the image into patches
        patches = image.reshape(num_patches_h, patch_size, num_patches_w, patch_size, c)
        patches = patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size, c)
        
        # Shuffle the patches
        np.random.shuffle(patches)
        
        # Reshape the shuffled patches back to the image shape
        shuffled_image = patches.reshape(num_patches_h, num_patches_w, patch_size, patch_size, c)
        shuffled_image = shuffled_image.swapaxes(1, 2).reshape(h, w, c)
        
        return shuffled_image
    
    
    def merge_shuffle_patches(self, image1, image2, patch_size=32):
        # Ensure the image dimensions are divisible by the patch size
        
        h, w, c = image1.shape
        
        def get_patch(image, patch_size):
            
            assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by the patch size."
            
            # Calculate the number of patches along each dimension
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            
            # Reshape the image into patches
            patches = image.reshape(num_patches_h, patch_size, num_patches_w, patch_size, c)
            patches = patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size, c)
            return patches, num_patches_h, num_patches_w
        
        # Shuffle the patches
        h, w, c = image1.shape
        patches1, num_patches_h, num_patches_w = get_patch(image1, patch_size)
    
        patches2, _, _                         = get_patch(image2, patch_size)
        np.random.shuffle(patches1)
        np.random.shuffle(patches2)
        
        num_patch = patches1.shape[0]
        
        ratio_concat = np.random.randint(int(num_patch * 0.3), int(0.8 * num_patch))
        
        patches1 = patches1[0:ratio_concat]
        patches2 = patches2[ratio_concat:num_patch]

        patches = np.concatenate([patches1, patches2], axis=0)
        
        index_patch = np.array(list(range(0, patches.shape[0])))
        
        np.random.shuffle(index_patch)
        patches = patches[index_patch]
        
        # Reshape the shuffled patches back to the image shape
        shuffled_image = patches.reshape(num_patches_h, num_patches_w, patch_size, patch_size, c)
        shuffled_image = shuffled_image.swapaxes(1, 2).reshape(h, w, c)
    
        return shuffled_image
    
    
    
    def merge_img(self, img1, img2):
        h, w = img1.shape[:2]
        if random.randint(0, 1):
            half1 = img1[:h//2]
            half2 = img2[:h//2]
            img = np.concatenate([half2, half1], axis=0)
        else:
            half1 = img1[:,:w//2]
            half2 = img2[:,:w//2]
            img = np.concatenate([half2, half1], axis=1)
        return img
    
    def break_img(self, img):
        h, w = img.shape[:2]
        r = random.uniform(0.7, 0.9)
        img = cv2.resize(img, None, fx=r, fy=r)
        img = cv2.resize(img, (w, h))
        return img
    
    
    def blur(self, img):
        ksize = random.choice([3, 5])
        img = cv2.blur(img, (ksize, ksize))  
        return img
    
    
    
    def random_mask_image(self, img, patch_size=[16, 16], prob=0.1):
    
        img_height, img_width, channels = img.shape

        # Calculate the number of patches along each dimension
        num_patches_y = img_height // patch_size[0]
        num_patches_x = img_width // patch_size[1]
        
        # Create a random mask for the patches
        patch_mask = np.random.uniform(size=(num_patches_y, num_patches_x)) < prob
        
        # black_white_mask = np.random.randint(0, 2, size=img.shape) * 255

        # Repeat the patch mask to match the size of the patches
        mask = np.repeat(np.repeat(patch_mask, patch_size[0], axis=0), patch_size[1], axis=1)

        # Make sure the mask is the same size as the image
        mask = np.pad(mask, ((0, img_height % patch_size[0]), (0, img_width % patch_size[1])), 'constant')
        
        # Apply the mask to the image
        img[mask] = random.randint(0, 1) #black_white_mask[mask]  # Assuming black mask

        return img
    
    
    def cv2_jpg(self, img):
        compress_val = random.choice(list(range(75, 100)))
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, compress_val]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        encimg = cv2.imdecode(encimg, 1)
        return encimg
    
    
    def pil_jpg(self, img):
        compress_val = random.choice(list(range(75, 100)))
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format="jpeg", quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img
    
    
    def gaussian_blur(self, img):
        
        def sample_continuous(s):
            if len(s) == 1:
                return s[0]
            if len(s) == 2:
                rg = s[1] - s[0]
                return random.random() * rg + s[0]
            raise ValueError("Length of iterable s should be 1 or 2.")
        
        sigma = sample_continuous([0.0, 1.0])
        gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
        gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
        gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
        return img
    
    
    def rotation(self, image):
        rotations = [90, 180, 270]
        angle = random.choice(rotations)
        if angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
        
    
    def simple_aug(self, img):
        
        if random.randint(0, 1):
            img = self.rotation(img)

        if random.randint(0, 1) == 0:
            img = cv2.flip(img, 1)
            
        # choices = np.random.choice(self.simple_choices, size=1, replace=False)
        # for i in choices:
        #     if i == 0:
        #         if random.randint(0, 1):
        #             img = self.cv2_jpg(img)
        #         else:
        #             img = self.pil_jpg(img)
        #     elif i == 1:
        #         img = self.gaussian_blur(img)
        #     elif i == 2:
        #         img = self.break_img(img)
                
        return img
    
    
    def __call__(self, img, aug_img=None):
        if random.uniform(0, 1) <= self.thresh:
            return img
    
        if random.randint(0, 1) == 0:
            img = self.simple_aug(img)
            
        return img
    
    
    
if __name__ == '__main__':
    
    class_ = DataAug()
    img = cv2.imread('debugs/000000000.jpg')
    img = class_.gaussian_blur(img)
    cv2.imwrite('debugs/000000001.jpg', img)