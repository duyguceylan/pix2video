import torch
import imageio
import numpy as np

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch_dtype=torch.float32

def warp(latents, warp_grid):
        '''
        warp_grid: b-by-h-by-w-by-2
        '''
        if len(latents) == 0:
            return [], []

        #resize warp grid
        warp_grid = torch.permute(warp_grid, (0,3,1,2))
        warp_grid = torch.nn.functional.interpolate(warp_grid, size=(latents[0].shape[2], latents[0].shape[3]))
        warp_grid = torch.permute(warp_grid, (0,2,3,1))
        
        mask = torch.ones_like(latents[0]).to(device)
        mask[:,:,0,0] = 0
        warped_mask = torch.nn.functional.grid_sample(mask, warp_grid.repeat(latents[0].shape[0],1,1,1), mode='nearest', padding_mode='zeros').to(device)
        
        warped_latents = []
        for i in range(len(latents)):
            warped_latent = torch.nn.functional.grid_sample(latents[i], warp_grid.repeat(latents[i].shape[0],1,1,1), mode='nearest').to(device)  
            warped_latents.append(warped_latent)

        return warped_latents, warped_mask


def get_intermediate_latents_from_warped_images(embedding_optimization, null_inversion, prev_img, prev_result, depth, org_prompt, edit_prompt, grid, W, H):
    #warp mask
    warp_grid = torch.permute(grid, (0,3,1,2))
    warp_grid = torch.nn.functional.interpolate(warp_grid, size=(H//8, W//8))
    warp_grid = torch.permute(warp_grid, (0,2,3,1))
    mask = torch.ones(1,4,H//8,W//8).to(device)
    mask[:,:,0,0] = 0
    warped_mask = torch.nn.functional.grid_sample(mask, warp_grid.repeat(mask.shape[0],1,1,1), mode='nearest', padding_mode="zeros").to(device) 
    imageio.imwrite('/home/ceylan/warped_mask.png', torch.permute(warped_mask[0,:3,:,:], (1,2,0)).detach().cpu().numpy())

    #warp the original and edited previous result
    warp_grid = torch.permute(grid, (0,3,1,2))
    warp_grid = torch.nn.functional.interpolate(warp_grid, size=(prev_img.shape[0], prev_img.shape[1]))
    warp_grid = torch.permute(warp_grid, (0,2,3,1))

    warped_edited_img = torch.nn.functional.grid_sample(torch.permute(torch.from_numpy(prev_result.astype(np.float32)).unsqueeze(dim=0),(0,3,1,2)).to(device), warp_grid, mode='bilinear', padding_mode="border").to(device) 
    warped_edited_img = torch.permute(warped_edited_img[0,:3,:,:], (1,2,0)).detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite('/home/ceylan/warped_edited.png', warped_edited_img)

    warped_org_img = torch.nn.functional.grid_sample(torch.permute(torch.from_numpy(prev_img.astype(np.float32)).unsqueeze(dim=0),(0,3,1,2)).to(device), warp_grid, mode='bilinear', padding_mode="border").to(device) 
    warped_org_img = torch.permute(warped_org_img[0,:3,:,:], (1,2,0)).detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite('/home/ceylan/warped_original.png', warped_org_img)

    enc_edited = embedding_optimization.image2latent(warped_edited_img)

    #invert the warped original image
    #null_inversion.init_prompt(org_prompt)
    #_, ddim_latents_org = null_inversion.ddim_inversion(warped_org_img, depth.to(device))
    #ddim_latents_org = ddim_latents_org[::-1]
    (image_gt, image_enc), x_t_org, uncond_emb_org, inverted_image, ddim_latents_org = null_inversion.invert_with_depth(    warped_org_img, depth, org_prompt, 
                                                                                                                            num_inner_steps=10)

    #edit the warped original image
    result, ddim_latents_edit = embedding_optimization.img2img_with_optimization(warped_org_img, depth, edit_prompt, 
                                                                                prev_img = enc_edited, 
                                                                                prev_latents = ddim_latents_org, 
                                                                                warp_grid=None, warp_mask=None, 
                                                                                optimize_cond=False, optimize_null=True,
                                                                                init_latent =  x_t_org.to(device), 
                                                                                uncond_embeddings=uncond_emb_org, inverted_noise=ddim_latents_org, 
                                                                                w_residual=0.0, w_encode=0.01,
                                                                                match_gradient=False)
        
    #result, ddim_latents_edit = embedding_optimization.img2img_cfg(warped_org_img, depth, edit_prompt, 
    #                                                                ref_latent=embedding_optimization.image2latent(warped_edited_img), 
    #                                                                init_latent = x_t_org.to(device), uncond_embeddings=uncond_emb_org)

    image = (result / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    imageio.imwrite('/home/ceylan/warped_original_edited.png', image)

    #invert the warped edited frame based on the warped original
    #null_inversion.init_prompt(edit_prompt)
    #_, ddim_latents_edit = null_inversion.ddim_inversion_ref(warped_edited_img, ddim_latents_org, ref_embedding=uncond_embeddings, depth=depth.to(device))
    #ddim_latents_edit = ddim_latents_edit[::-1]
        
    #_, ddim_latents_edit_no_opt = null_inversion.ddim_inversion(warped_edited_img, depth.to(device))
    #ddim_latents_edit_no_opt = ddim_latents_edit_no_opt[::-1]
    #    
    ##encode the warped image
    #prev_result = ddim_latents_edit[-1]

    #residuals
    prev_latents = []
    for j in range(len(ddim_latents_edit)):
        prev_latents.append(ddim_latents_edit[j]- ddim_latents_org[j])
        print(torch.nn.functional.mse_loss(ddim_latents_edit[j], ddim_latents_org[j]))
        #print(torch.nn.functional.mse_loss(ddim_latents_edit_no_opt[j], ddim_latents_org[j]))

    return ddim_latents_org, prev_latents, enc_edited, warped_mask
