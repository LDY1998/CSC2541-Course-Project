import pickle, numpy as np
import torch
import utils


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
        self.obsnorm_mean = torch.Tensor(policy_params['obsnorm']['Standardizer']['mean_1_D'])
        self.obsnorm_meansq = torch.Tensor(policy_params['obsnorm']['Standardizer']['meansq_1_D'])
        self.obsnorm_stdev = torch.Tensor(np.sqrt(np.maximum(0, self.obsnorm_meansq - np.square(self.obsnorm_mean))))

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
        out_layer = torch.nn.Linear(W.shape[0], W.shape[1])
        
        with torch.no_grad():
            out_layer.weight.copy_(torch.transpose(W, 0, 1))
            out_layer.bias.copy_(torch.squeeze(b))
        self.out_layer = out_layer

    def forward(self, x):
        x = (x - self.obsnorm_mean) / (self.obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class
        x = x.to(torch.float32)
        for layer in self.layers:

            # print(f"W type: {layer.weight.dtype} \n b type: {layer.bias.dtype}  x type: {x.dtype}\n\n")
            x = layer(x)
            x = apply_nonlin(x, self.nonlin_type)
        x = self.out_layer(x)
        return x

# for testing
if __name__ == "__main__":
    policy = ExpertPolicy("experts/Hopper-v2.pkl")