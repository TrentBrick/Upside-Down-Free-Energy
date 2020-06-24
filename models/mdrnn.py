"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

def gmm_loss(latent_target, mus, sigmas, logpi): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of next_latents under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args next_latents: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs, fs) torch tensor #NOTE: added the gs dimension here. 
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(next_latents) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            next_latents[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    """
    latent_target = latent_target.unsqueeze(-2) # to acccount for the gaussian dimension. 
    normal_dist = Normal(mus, sigmas) # for every gaussian in each latent dimension. 
    #print('MDRNN TEST. WHAT ARE THE DIMENSIONS OF THESE????', logpi.shape, mus.shape, latent_target.shape)
    g_log_probs = logpi + normal_dist.log_prob(latent_target) # how far off are the next obs? 
    # sum across the gaussians, need to do so in log space: 
    log_loss = - torch.logsumexp(g_log_probs, dim=-2) # now have bs1, bs2, fs. all of these are different predictions to take the mean of.   
    #print(torch.exp(logpi))
    #print('gmm loss', latent_target.shape, mus.shape, sigmas.shape, g_log_probs.shape, log_loss.shape)
    return torch.mean(log_loss)

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians, conditional):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.conditional = conditional 

        self.gmm_linear = nn.Linear(
            hiddens, (3 * latents * gaussians) + 2 ) # each latent for each gaussian has its own pi, mu and sigma. another 2 for reward and termination. 

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians, conditional=True,
                        no_lstm=True):
        super().__init__(latents, actions, hiddens, gaussians, conditional)
        
        self.no_lstm = no_lstm
        if no_lstm: 
            self.forward1 = nn.Linear(latents + actions+1, 128)
            self.forward2 = nn.Linear(128, 256)
            self.forward3 = nn.Linear(256, 128)
            self.forward4 = nn.Linear(128, latents+1)
        else:
            if self.conditional: 
                self.rnn = nn.LSTM(latents + actions+1, hiddens, batch_first=True)
            else: 
                self.rnn = nn.LSTM(latents + actions, hiddens, batch_first=True)

    def forward(self, actions, latents, r, last_hidden=None): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (BSIZE, SEQ_LEN, ACTION_SIZE) torch tensor
        :args latents: (BSIZE, SEQ_LEN, LATENT_SIZE) torch tensor
        :args r: (BSIZE, SEQ_LEN, 1) torch tensor, rewards 

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (BSIZE, SEQ_LEN, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, SEQ_LEN, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, SEQ_LEN, N_GAUSS, LSIZE) torch tensor
            - rs: (BSIZE, SEQ_LEN) torch tensor
            - ds: (BSIZE, SEQ_LEN) torch tensor
        """
        batch_size, seq_len = actions.size(0), actions.size(1)
        #r = r.unsqueeze(1)

        if self.no_lstm:
            ins = torch.cat([actions, latents, r], dim=-1)

            outs = self.forward1(ins)
            outs = self.forward2(outs)
            outs = self.forward3(outs)
            outs = self.forward4(outs)

            #print('returning from forward model!', outs.shape)

            if len(outs.shape) ==2:
                return outs[:,:self.latents], torch.zeros((5,5,5)), torch.zeros((5,5,5)), outs[:,-1], torch.zeros((5,5,5)), (torch.zeros((5,5,5)),torch.zeros((5,5,5)))


            if last_hidden:
                return outs[:,:,:self.latents], torch.zeros((5,5,5)), torch.zeros((5,5,5)), outs[:,:,-1], torch.zeros((5,5,5)), (torch.zeros((5,5,5)),torch.zeros((5,5,5)))
            else: 
                return outs[:,:,:self.latents], torch.zeros((5,5,5)), torch.zeros((5,5,5)), outs[:,:,-1], torch.zeros((5,5,5))

        else: 
            #print(actions.shape, latents.shape)
            if self.conditional: 
                ins = torch.cat([actions, latents, r], dim=-1)
            else: 
                ins = torch.cat([actions, latents], dim=-1)

            if last_hidden:
                outs, last_hidden = self.rnn(ins, last_hidden)
            else: 
                outs, _ = self.rnn(ins)
            gmm_outs = self.gmm_linear(outs)

            stride = self.gaussians * self.latents # number of predictions per element.

            mus = gmm_outs[:, :, :stride]
            mus = mus.view(batch_size, seq_len, self.gaussians, self.latents)

            sigmas = gmm_outs[:, :, stride:2 * stride]
            sigmas = sigmas.view(batch_size, seq_len, self.gaussians, self.latents)
            sigmas = torch.exp(sigmas) # assuming they were logged before? 

            pi = gmm_outs[:, :, 2 * stride: 3 * stride ]
            pi = pi.view(batch_size, seq_len, self.gaussians, self.latents)
            logpi = f.log_softmax(pi, dim=-2) #NOTE:Check this is working as it should be!! and update in the Cell version too below!!

            # TODO: enable removing these channels because they are taking up space!
            rs = gmm_outs[:, :, -2]

            ds = gmm_outs[:, :, -1]

            if last_hidden:
                return mus, sigmas, logpi, rs, ds, last_hidden
            else: 
                return mus, sigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians, conditional=True):
        super().__init__(latents, actions, hiddens, gaussians, conditional)
        if self.conditional: 
            self.rnn = nn.LSTMCell(latents + actions+1, hiddens)
        else: 
            self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden, r): # pylint: disable=arguments-differ
        """ ONE STEP forward.

        :args actions: (BSIZE, Action SIZE) torch tensor
        :args latents: (BSIZE, Latent SIZE) torch tensor
        :args hidden: (BSIZE, Recurrent hidden SIZE) torch tensor
        :args r: (BSIZE, 1) torch tensor, rewards 

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, Latent SIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, Latent SIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        #r = r.unsqueeze(1)
        if self.conditional: 
            in_al = torch.cat([action, latent, r], dim=1)
        else: 
            in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        log_sigmas = out_full[:, stride:2 * stride]
        log_sigmas = log_sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(log_sigmas)

        pi = out_full[:, 2 * stride:3 * stride]
        pi = pi.view(-1, self.gaussians, self.latents)
        logpi = f.log_softmax(pi, dim=-2)

        r = out_full[:, -2]
        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
