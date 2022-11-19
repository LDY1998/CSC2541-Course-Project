import pickle, numpy as np
import tensorflow as tf
import torch
import utils

def load_policy(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    def build_policy(obs_bo):
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        def apply_nonlin(x):
            if nonlin_type == 'lrelu':
                # l = torch.relu(x)
                # return tf_util.lrelu(x, leak=.01) # openai/imitation nn.py:233
                return utils.lrelu(x, leak=.01)
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
                # return torch.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)

        # obs_bo = torch.zeros_like(obsnorm_mean)
        normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation
        print("norm obs: ", normedobs_bo.shape)

        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            curr_activations_bd = apply_nonlin(torch.matmul(curr_activations_bd, W) + b)

        # Output layer
        W, b = read_layer(policy_params['out'])
        # output_bo = tf.matmul(curr_activations_bd, W) + b
        output_bo = torch.matmul(curr_activations_bd, W) + b
        return output_bo

    obs_bo = tf.placeholder(tf.float32, [None, None])
    # obs_bo = torch.ones()
    a_ba = build_policy(obs_bo)
    # a_ba = build_policy()
    policy_fn = utils.function([obs_bo], a_ba)
    return policy_fn

def load_policy_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}
    return policy_params, nonlin_type

def apply_nonlin(x, nonlin_type):
        if nonlin_type == 'lrelu':
            # l = torch.relu(x)
            # return tf_util.lrelu(x, leak=.01) # openai/imitation nn.py:233
            return utils.lrelu(x, leak=.01)
        elif nonlin_type == 'tanh':
            return torch.tanh(x)
            # return torch.tanh(x)
        else:
            raise NotImplementedError(nonlin_type)

class ExpertPolicy(torch.nn.Module):
    def __init__(self, filename) -> None:
        super().__init__()
        policy_params, self.nonlin_type = load_policy_file(filename)
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        self.obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        self.obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        self.obsnorm_stdev = np.sqrt(np.maximum(0, self.obsnorm_meansq - np.square(self.obsnorm_mean)))

        self.layers = []
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            W, b = l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)
            return torch.Tensor(W), torch.Tensor(b)
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            
            layer = torch.nn.Linear(W.shape[0], W.shape[1])
            with torch.no_grad():
                layer.weight.copy_(torch.transpose(W, 0, 1))
                layer.bias.copy_(torch.squeeze(b))
            self.layers.append(layer)
        
        W, b = read_layer(policy_params['out'])
        layer = torch.nn.Linear(W.shape[0], W.shape[1])
        
        with torch.no_grad():
            layer.weight.copy_(torch.transpose(W, 0, 1))
            layer.bias.copy_(torch.squeeze(b))
        self.layers.append(layer)


    def forward(self, x):
        x = (x - self.obsnorm_mean) / (self.obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class
        x = x.to(torch.float32)
        for layer in self.layers:

            print(f"W type: {layer.weight.dtype} \n b type: {layer.bias.dtype}  x type: {x.dtype}\n\n")
            x = layer(x)
            x = apply_nonlin(x, self.nonlin_type)
        return x

if __name__ == "__main__":
    policy = ExpertPolicy("experts/Hopper-v2.pkl")
    # load_policy("experts/Hopper-v2.pkl")