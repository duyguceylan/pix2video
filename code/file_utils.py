import torch
import numpy as np
import imageio
import PIL
from PIL import Image
import os
from einops import repeat, rearrange

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_data_paths(folder_path, output_path, noFrames, reverse=False):
    image_paths = []
    depth_paths = []
    noise_paths = []
    output_paths = []
    frame_numbers = []

    for i in range(noFrames):
        f = '%06d.png' % (i)

        image_paths.append(os.path.join(folder_path, f)) 
        depth_paths.append(os.path.join(folder_path, f[:-3]+'npy')) 
        noise_paths.append(os.path.join(folder_path, f[:-4]))
        output_paths.append(os.path.join(output_path, f))
        frame_numbers.append(i)
    
    if reverse:
        image_paths = image_paths[::-1]
        depth_paths = depth_paths[::-1]
        noise_paths = noise_paths[::-1]
        output_paths = output_paths[::-1]
        frame_numbers = frame_numbers[::-1]
    
    return image_paths, depth_paths, noise_paths, output_paths, frame_numbers

def load_512(image_path, output_path='', left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    
    #h, w, c = image.shape
    #if h < w:
    #    offset = (w - h) // 2
    #    image = image[:, offset:offset + h]
    #elif w < h:
    #    offset = (h - w) // 2
    #    image = image[offset:offset + w]
    
    h, w, c = image.shape

    if False:
    #if h % 64 != 0 or w % 64 != 0:

        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        
        while w > 512 or h > 512:
            w /= 2
            h /= 2

        image = Image.fromarray(image)

        image = image.resize((int(w), int(h)), resample=PIL.Image.LANCZOS)

        if output_path != '':
            imageio.imwrite(output_path, np.array(image))
        else:
            imageio.imwrite(output_path, np.array(image))

        image = np.array(image.convert("RGB"))

    image = Image.fromarray(image).resize((512, 512))
    image = np.array(image.convert("RGB"))

    return image

def load_512_crop(image_path, output_path='', left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    
    h, w, c = image.shape
    if h < w:
        offset = (w - h) #// 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w)//2
        #offset -= 200
        image = image[offset:offset + w]
        #image = image[300:offset+300]
    
    h, w, c = image.shape

    image = Image.fromarray(image)

    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)

    if output_path != '':
        imageio.imwrite(output_path, np.array(image))
    else:
        imageio.imwrite(output_path, np.array(image))

    image = np.array(image.convert("RGB"))

    #image = Image.fromarray(image).resize((1024, 1024))
    #image = np.array(image.convert("RGB"))

    return image

def load_inversion_epsilon( invert_path, input_image, depth, prompt, null_inversion, 
                            warp_grid=None, warp_mask=None,
                            num_inner_steps=10):
    if os.path.exists(invert_path+'_epsilon.npy'):
        inverted_noise = np.load(invert_path + '_inverted_latent.npy')
        uncond_embeddings = np.load(invert_path + '_uncond_emb.npy')
        epsilon_list = np.load(invert_path + '_epsilon.npy')
    else:
        x_t, uncond_embeddings, inverted_image, inverted_noise, epsilon_list = null_inversion.invert_with_depth(   input_image, 
                                                                                                                    depth, 
                                                                                                                    prompt, 
                                                                                                                    num_inner_steps = num_inner_steps, 
                                                                                                                    verbose=True)
        inverted_noise = torch.stack(inverted_noise).detach().cpu().numpy()
        uncond_embeddings = torch.stack(uncond_embeddings).detach().cpu().numpy()
        epsilon_list = torch.stack(epsilon_list).detach().cpu().numpy()
        with open(invert_path + '_inverted_latent.npy', 'wb') as of:
            np.save(of, inverted_noise)
        with open(invert_path + '_uncond_emb.npy', 'wb') as of:
            np.save(of, uncond_embeddings)
        with open(invert_path + '_epsilon.npy', 'wb') as of:
            np.save(of, epsilon_list)

    return inverted_noise, uncond_embeddings, epsilon_list

def load_inversion( invert_path, input_image, depth, prompt, null_inversion, 
                    ref_latents = None, warp_grid=None, warp_mask=None,
                    num_inner_steps=10):
    if os.path.exists(invert_path+'_inverted_latent.npy'):
        inverted_noise = np.load(invert_path + '_inverted_latent.npy')
        uncond_embeddings = np.load(invert_path + '_uncond_emb.npy')
    else:
        if ref_latents is not None:
            (image_gt, image_enc), x_t, uncond_embeddings, inverted_image, inverted_noise = null_inversion.invert_with_depth_with_ref(  input_image, 
                                                                                                                                        depth, 
                                                                                                                                        prompt, 
                                                                                                                                        ref_latents, 
                                                                                                                                        warp_grid, 
                                                                                                                                        warp_mask,
                                                                                                                                        num_inner_steps=num_inner_steps, 
                                                                                                                                        verbose=True)
        else:
            (image_gt, image_enc), x_t, uncond_embeddings, inverted_image, inverted_noise = null_inversion.invert_with_depth(   input_image, 
                                                                                                                                depth, 
                                                                                                                                prompt, 
                                                                                                                                num_inner_steps = num_inner_steps, 
                                                                                                                                verbose=True)
        inverted_noise = torch.stack(inverted_noise).detach().cpu().numpy()
        uncond_embeddings = torch.stack(uncond_embeddings).detach().cpu().numpy()
        with open(invert_path + '_inverted_latent.npy', 'wb') as of:
            np.save(of, inverted_noise)
        with open(invert_path + '_uncond_emb.npy', 'wb') as of:
            np.save(of, uncond_embeddings)

    return inverted_noise, uncond_embeddings

def load_depth(model, depth_path, input_image, W=512, H=512, inverted=False):

    if os.path.exists(depth_path):
        depth = np.load(depth_path)

    else:
        pixel_values = model.feature_extractor(images=[input_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device)
        depth = model.depth_estimator(pixel_values).predicted_depth
        depth = depth.permute((1,2,0)).squeeze(dim=-1)
        depth = depth.detach().cpu().numpy()
        with open(depth_path, 'wb') as of:
            np.save(of, depth)
        inverted=False
    
    indices = np.where(depth != -1)
    bg_indices = np.where(depth == -1)
    d_values = depth[indices[0], indices[1]]
    min_d = np.min(d_values)
    max_d = np.max(d_values)

    if inverted:
        if len(bg_indices[0] > 0):
            depth[bg_indices] = max_d + 10
            max_d = max_d + 10
        depth = -depth
    else:
        if len(bg_indices[0] > 0):
            depth[bg_indices] = min_d - 10
            min_d = min_d - 10

    depth = torch.from_numpy(depth)
    depth = rearrange(depth, 'h w -> 1 1 h w')

    depth_min, depth_max = torch.amin(depth, dim=[1, 2, 3], keepdim=True), torch.amax(depth, dim=[1, 2, 3],
                                                                                           keepdim=True)
    depth = torch.nn.functional.interpolate(
                depth,
                size=(H//8,W//8),
                mode="bilinear",
                align_corners=False,
    )
    depth_min, depth_max = torch.amin(depth, dim=[1, 2, 3], keepdim=True), torch.amax(depth, dim=[1, 2, 3],
                                                                                           keepdim=True)
    depth = 2. * (depth - depth_min) / (depth_max - depth_min) - 1.
    depth = depth.to(device).type(torch.FloatTensor)

    return depth
