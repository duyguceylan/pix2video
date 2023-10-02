import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim.adam import Adam

import torch.nn.functional as nnf
from mydiffusers import DDIMScheduler
from mydiffusers.models.embeddings import TimestepEmbedding, Timesteps
from PIL import Image
import imageio
from tqdm import tqdm
import os
from file_utils import *
from custom_utils import *
from pytorch_lightning import seed_everything
from random import randrange
import time

class VideoDataset(Dataset):
    """ Dataset for loading the Human Alloy data. In this case, images are 
    chosen within a video.
    """

    def __init__(self, image_model, image_array, depth_array, data_path, inversion_path, device, is_reverse=False):
        self.image_array = image_array
        self.model = image_model
        self.device = device
        
        self.image_latents = []
        self.depth = []
        self.inversions = []

        self.data_path = data_path
        self.inversion_path = inversion_path

        for i in range(len(image_array)):
            self.image_latents.append(self.image2latent(image_array[i]))
            self.depth.append(depth_array[i].to(self.device))

            inv_path = os.path.join(inversion_path, '%06d.pt' % i)
            if os.path.exists(inv_path):
                inv = torch.load(inv_path)
                if len(inv) > 1:
                    inv = inv[-1]
                self.inversions.append(inv.to(self.device))

        H = image_array[0].shape[0]
        W = image_array[0].shape[1]
        
        self.no_frames = len(image_array)
        self.local_neighbor_indices = [-2, -1, 1, 2]

        self.keyframe_index = 0
        self.is_reverse = is_reverse
        if is_reverse:
            self.keyframe_index = self.no_frames-1
            print('keyframe_index:%d\n' % self.keyframe_index)

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).type(torch.FloatTensor)  / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    def __getitem__(self, index):
        neighbor_index = index-1
        if neighbor_index < 0:
            neighbor_index += self.no_frames
        elif neighbor_index >= self.no_frames:
            neighbor_index -= self.no_frames

        return {'latent' : self.image_latents[index], 
                'depth': self.depth[index],
                'frame' : torch.tensor(index, device=self.device),
                'neighbor': self.image_latents[neighbor_index],
                'neighbor_depth': self.depth[neighbor_index],
                'neighbor_frame': torch.tensor(neighbor_index, device=self.device)}

    def __len__(self):
        return (len(self.image_latents))

    def toval(self, epoch):
        pass

    def totrain(self, epoch):
        pass

class VideoModel(torch.nn.Module):
    
    def __init__(   self, model, NUM_DDIM_STEPS, GUIDANCE_SCALE, output_path, org_prompt, edit_prompt, no_frames,
                    use_keyframe_only=False,  
                    classifier_guidance_flag = False, per_frame_classifier_guidance = False, 
                    use_self_attention_injection=True, 
                    target_index = -1, device='cuda:0'):
        '''
        use_keyframe_only: if set to True, we will attend only to an anchor frame. if set to False, we will attend to an anchor frame and the previous frame
        classifier_guidance_flag: if set to True, we will perform the latent update. if set to False, we won't perform the latent update
        per_frame_classifier_guidance: if set to True, for each latent update, we will use the previous frame's latent at the corresponding diffusion step
        target_index: if it is not -1, for each latent update,  we will use the previous frame's latent at the target_index diffusion step
        '''
        super().__init__()

        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.GUIDANCE_SCALE = GUIDANCE_SCALE
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.device = device
        self.output_path = output_path
        self.org_prompt = org_prompt
        self.edit_prompt = edit_prompt
        self.use_keyframe_only = use_keyframe_only
        self.no_frames = no_frames
        self.classifier_guidance_flag = classifier_guidance_flag
        self.per_frame_classifier_guidance = per_frame_classifier_guidance
        self.use_self_attention_injection = use_self_attention_injection
        self.target_index = target_index

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

        self.finetune_layers = []

        self.unet = create_custom_diffusion(self.model.unet, self.finetune_layers)

        self.init_prompt(self.org_prompt)

    def auto_corr_loss(self, x, random_shift=True):
        B,C,H,W = x.shape
        assert B==1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None,:,:]
            while True:
                if random_shift: roll_amount = randrange(noise.shape[2]//2)
                else: roll_amount = 1
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=2)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    def kl_divergence(self, x):
        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var+1e-7)

    def ddim_loop(self, latent, depth, self_attn_query_features_uncond_list_prev = [], self_attn_query_features_cond_list_prev = [], latents_x0_list_prev = []):
        latents_x0_list = []
        latent_list = []

        latent = latent.clone().detach()
        self_attn_query_features_uncond_list = []
        self_attn_query_features_cond_list = []

        latent_list.append(latent)

        for i in range(self.NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]

            self_attn_query_feat_uncond = None
            self_attn_query_feat_cond = None
            
            if len(self_attn_query_features_uncond_list_prev) > 0:
                self_attn_query_feat_uncond = self_attn_query_features_uncond_list_prev[i]
            if len(self_attn_query_features_cond_list_prev) > 0:
                self_attn_query_feat_cond = self_attn_query_features_cond_list_prev[i]

            latent, latent_x0, self_attn_query_features_uncond, self_attn_query_features_cond = self.get_noise_pred(   latent, t, depth=depth,
                                                                                                                self_attn_query_feat_uncond =self_attn_query_feat_uncond,
                                                                                                                self_attn_query_feat_cond = self_attn_query_feat_cond,
                                                                                                                is_forward=True)

            self_attn_query_features_uncond_list.append(self_attn_query_features_uncond)
            self_attn_query_features_cond_list.append(self_attn_query_features_cond)
            latents_x0_list.append(latent_x0)
            latent_list.append(latent)

        return latent_list, latents_x0_list, self_attn_query_features_uncond_list, self_attn_query_features_cond_list

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        batch_size = len(prompt)

        uncond_input = self.model.tokenizer(
            [""]* batch_size, padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        #self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.context = [uncond_embeddings, text_embeddings]
        self.prompt = prompt

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        image = (image).clamp(-1, 1)
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample, pred_original_sample

    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

        return next_sample, next_original_sample

    #duygu: ignore latents_x0_list_prev and warp_grid
    def img2img(self, latent, depth, init_latents = None, prompt=[], 
                self_attn_query_features_uncond_list_prev = [], 
                self_attn_query_features_cond_list_prev = [],
                latents_x0_list_prev = [],
                start_step = 0):
        if len(prompt) > 0:
            self.init_prompt(prompt)

        noise = torch.randn_like(latent).to(self.device)

        # get latents
        if init_latents is None:
            t = self.model.scheduler.timesteps[start_step]
            init_latents = self.model.scheduler.add_noise(latent, noise, t)

        # self_attn_query_features_uncond_list and self_attn_query_features_cond_list will contain the features of the current frame
        result, self_attn_query_features_uncond_list, self_attn_query_features_cond_list, latents_x0_list = self.decode(init_latents, depth=depth,
                                                                                                                        self_attn_query_features_uncond_list_prev = self_attn_query_features_uncond_list_prev,
                                                                                                                        self_attn_query_features_cond_list_prev = self_attn_query_features_cond_list_prev,
                                                                                                                        latents_x0_list_prev = latents_x0_list_prev,
                                                                                                                        start_step=start_step,
                                                                                                                        return_type = 'np')
        return result, self_attn_query_features_uncond_list, self_attn_query_features_cond_list, latents_x0_list

    def get_noise_pred_single(self, latents, t, context, depth=None):
        if depth is not None:
            noise_pred = self.forward(torch.cat((latents, depth), dim=1), t, encoder_hidden_states=context)["sample"]
        else:
            noise_pred = self.forward(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, context=None, depth=None, self_attn_query_feat_uncond = None, self_attn_query_feat_cond = None, 
                       pred_x0=None, is_forward=False):
        latents_org = None

        if context is None:
            context = torch.cat(self.context)

        uncond_embeddings, cond_embeddings = torch.chunk(context, 2)

        guidance_scale = self.GUIDANCE_SCALE
        if is_forward:
            guidance_scale = 1.0

        if depth is not None:
            latents_input = torch.cat((latents, depth), dim=1)
        else:
            latents_input = latents

        #duygu: ignore warp_grid
        output = self.forward(latents_input, t, encoder_hidden_states=uncond_embeddings, self_attn_query_features_cond=self_attn_query_feat_uncond)
        
        noise_pred_uncond = output["sample"]
        self_attn_features_uncond = output["self_attn_query_features"]

        output = self.forward(latents_input, t, encoder_hidden_states=cond_embeddings, self_attn_query_features_cond=self_attn_query_feat_cond)

        noise_pred_cond = output["sample"]
        self_attn_features_cond = output["self_attn_query_features"]

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        if is_forward:
            latents, latents_org = self.next_step(noise_pred, t, latents)
        else:
            latents, latents_org = self.prev_step(noise_pred, t, latents)


        return latents, latents_org, self_attn_features_uncond, self_attn_features_cond

    def decode( self, latents, depth=None,
                self_attn_query_features_uncond_list_prev = [], 
                self_attn_query_features_cond_list_prev = [], 
                latents_x0_list_prev = [],
                start_step = 0,
                return_type = 'np'):

        latent_cur = latents
        self_attn_query_features_uncond_list = []
        self_attn_query_features_cond_list = []
        latents_x0_list = []

        per_step_weight = 100.0
        for i in range(start_step, self.NUM_DDIM_STEPS):

            if i>25:
                per_step_weight = 0

            self_attn_query_feat_uncond = None
            self_attn_query_feat_cond = None
            latent_x0_prev = None

            if len(self_attn_query_features_uncond_list_prev) > 0:
                self_attn_query_feat_uncond = self_attn_query_features_uncond_list_prev[i-start_step]
            if len(self_attn_query_features_cond_list_prev) > 0:
                self_attn_query_feat_cond = self_attn_query_features_cond_list_prev[i-start_step]
            
            if len(latents_x0_list_prev) > 0:
                if self.per_frame_classifier_guidance:
                    latent_x0_prev = latents_x0_list_prev[i-start_step]
                else:
                    latent_x0_prev = latents_x0_list_prev[self.target_index]

            t = self.model.scheduler.timesteps[i]

            uncond_emb = self.context[0] 
            
            cond_embeddings = self.context[1]

            uncond_emb = uncond_emb.repeat(latents.shape[0], 1, 1)
            cond_embeddings = cond_embeddings.repeat(latents.shape[0], 1, 1)

            context = torch.cat((uncond_emb,cond_embeddings ))
            latent_cur, latent_x0, self_attn_query_features_uncond, self_attn_query_features_cond = self.get_noise_pred(latent_cur, t, 
                                                                                                                        context=context, 
                                                                                                                        depth=depth,
                                                                                                                        self_attn_query_feat_uncond =self_attn_query_feat_uncond,
                                                                                                                        self_attn_query_feat_cond = self_attn_query_feat_cond,
                                                                                                                        pred_x0 = latent_x0_prev)

            #self_attn_query_features_uncond and self_attn_query_features_cond are features from the current diffusion step
            #self_attn_query_features_uncond_list and self_attn_query_features_cond_list hold features from all diffusion steps
            self_attn_query_features_uncond_list.append(self_attn_query_features_uncond)
            self_attn_query_features_cond_list.append(self_attn_query_features_cond)
            
            ## classifier-based guidance
            target_index = i-start_step if self.classifier_guidance_flag and self.per_frame_classifier_guidance else self.target_index
            if self.classifier_guidance_flag and len(latents_x0_list_prev) > 0:
                target = latents_x0_list_prev[target_index]

                with torch.enable_grad():
                    dummy_pred = latent_x0.clone().detach()
                    
                    dummy_pred = dummy_pred.requires_grad_(requires_grad=True)
                    loss = per_step_weight * torch.nn.functional.mse_loss(target, dummy_pred)
                    loss.backward()
                    latent_cur = latent_cur + dummy_pred.grad.clone() * -1.

            latents_x0_list.append(latent_x0)

        image = self.latent2image(latent_cur, return_type=return_type)
        
        return image, self_attn_query_features_uncond_list, self_attn_query_features_cond_list, latents_x0_list

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        self_attn_query_features_cond=None,
        return_dict: bool = True,
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        
        #unet now refers to the unet_2d_condition.py class under mydiffusers/models
        return self.unet(sample, timestep, encoder_hidden_states, self_attn_query_features_cond=self_attn_query_features_cond)

    def perform_inversion(self, dataset):

        if len(dataset.inversions) > 0:
            print('Inversion already loaded!')
            return 

        os.makedirs(dataset.inversion_path, exist_ok=True)

        self.init_prompt(self.org_prompt)

        with torch.no_grad():
            for i in range(self.no_frames):
                b = i
                if dataset.is_reverse:
                    b = len(dataset.image_latents) - i -1

                latent = dataset.image_latents[b]
                depth = dataset.depth[b]

                #do inversion
                latent_noisy_list, _, _, _ = self.ddim_loop(   latent, depth)
                
                latent_noisy = latent_noisy_list[-1]

                dataset.inversions.append(latent_noisy)

                torch.save(latent_noisy_list, os.path.join(dataset.inversion_path, '%06d.pt' % b))     

        if dataset.is_reverse:
            dataset.inversions = dataset.inversions[::-1]

    def test(self, dataset, output_path, prompt):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        test_count = 0

        #self_attn_query_features_uncond_list and self_attn_query_features_cond_list will hold the features obtained from the current frame
        #self_attn_query_features_uncond_list_prev and self_attn_query_features_cond_list_prev hold the features obtained from the previous frame
        #self_attn_query_features_uncond_list_first and self_attn_query_features_cond_list_first hold the features obtained from frame 0
        self_attn_query_features_uncond_list = []
        self_attn_query_features_cond_list = []

        self_attn_query_features_uncond_list_first = []
        self_attn_query_features_cond_list_first = []

        self_attn_query_features_uncond_list_prev = []
        self_attn_query_features_cond_list_prev = []

        prev_latents_x0_list = []

        start_step = 0

        self.init_prompt(prompt)

        with torch.no_grad():
            for i in range(self.no_frames):
                t = time.process_time()

                b = i
                if dataset.is_reverse:
                    b = len(dataset.image_latents) - i -1

                latent = dataset.image_latents[b]
                depth = dataset.depth[b]

                if i==0:
                    latent_noisy = torch.randn_like(latent).to(latent.device)
                
                if len(dataset.inversions) > 0 and start_step == 0:
                    latent_noisy = dataset.inversions[b]
                else:
                    latent_noisy = None

                uncond_emb = None

                #duygu:
                #self_attn_query_features_uncond_list_cur and self_attn_query_features_cond_list_cur will contain the features of the current frame
                #you can ignore latents_x0_list_prev and warp_grid
                img, self_attn_query_features_uncond_list_cur, self_attn_query_features_cond_list_cur, latents_x0_list = self.img2img(  latent, depth,
                                                                                                                                        init_latents = latent_noisy,
                                                                                                                                        self_attn_query_features_uncond_list_prev = self_attn_query_features_uncond_list,
                                                                                                                                        self_attn_query_features_cond_list_prev = self_attn_query_features_cond_list,
                                                                                                                                        latents_x0_list_prev = prev_latents_x0_list,
                                                                                                                                        start_step=start_step)
                #duygu: in the below code, we append the features of frame 0 with previous
                #for now you can just update the previous features to be equal to current features
                elapsed_time = time.process_time() - t
                print('elapsed time: ', elapsed_time)
                prev_latents_x0_list = latents_x0_list
                if i == 0 and self.use_self_attention_injection:
                    self_attn_query_features_uncond_list_first = []
                    self_attn_query_features_cond_list_first = []

                    for s in range(len(self_attn_query_features_uncond_list_cur)):
                        self_attn_query_features_uncond_list_first.append([])
                        self_attn_query_features_cond_list_first.append([])

                        for j in range(len(self_attn_query_features_uncond_list_cur[s])):
                            self_attn_query_features_uncond_list_first[s].append(self_attn_query_features_uncond_list_cur[s][j])
                            self_attn_query_features_cond_list_first[s].append(self_attn_query_features_cond_list_cur[s][j])
                
                if self.use_self_attention_injection:
                    if self.use_keyframe_only:
                        self_attn_query_features_uncond_list = self_attn_query_features_uncond_list_first
                        self_attn_query_features_cond_list = self_attn_query_features_cond_list_first
                    else:
                        self_attn_query_features_uncond_list = []
                        self_attn_query_features_cond_list = []
                        for s in range(len(self_attn_query_features_uncond_list_cur)):
                            self_attn_query_features_uncond_list.append([])
                            self_attn_query_features_cond_list.append([])
                            for j in range(len(self_attn_query_features_uncond_list_cur[s])):
                                self_attn_query_features_uncond_list[s].append(torch.cat([self_attn_query_features_uncond_list_first[s][j], self_attn_query_features_uncond_list_cur[s][j]], dim=1))
                                self_attn_query_features_cond_list[s].append(torch.cat([self_attn_query_features_cond_list_first[s][j], self_attn_query_features_cond_list_cur[s][j]], dim=1))
                
                image = Image.fromarray(img[0])
                image = np.array(image.convert("RGB"))
                imageio.imwrite(os.path.join(output_path, '%06d.png' % (b)), image)
