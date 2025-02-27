import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
          input_size: int
          hidden_sizes: list of int
          output_size: int
        """
        super(MLP, self).__init__()
        
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.linears.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_sizes[0]))
        
        # Add additional hidden layers if specified
        previous_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            self.linears.append(torch.nn.Linear(previous_size, size))
            self.batch_norms.append(torch.nn.BatchNorm1d(size))
            previous_size = size
        
        self.output = torch.nn.Linear(previous_size, output_size)
  
    def forward(self, x):
        for linear, batch_norm in zip(self.linears, self.batch_norms):
            x = linear(x)
            x = batch_norm(x)
            x = F.relu(x)

        return self.output(x)


class MLPGenerator(nn.Module):
    def __init__(self, input_size, hidden_sizes, out_vectors, outdim=3):
        """
          Used to generate output strokes. It automatically reshapes output.

          input_size: int
          hidden_sizes: list of int
          out_vectors: int
                       number of output vectors
          outdim: int
                  translational dims of each output vector
        """
        super(MLPGenerator, self).__init__()

        self.out_vectors = out_vectors
        self.outdim = outdim
        output_size = self.out_vectors*self.outdim
        
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.linears.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_sizes[0]))
        
        # Add additional hidden layers if specified
        previous_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            self.linears.append(torch.nn.Linear(previous_size, size))
            self.batch_norms.append(torch.nn.BatchNorm1d(size))
            previous_size = size
        
        self.output = torch.nn.Linear(previous_size, output_size)
  
    def forward(self, x):
        B, _ = x.shape

        for linear, batch_norm in zip(self.linears, self.batch_norms):
            x = linear(x)
            x = batch_norm(x)
            x = F.relu(x)

        x = self.output(x)

        out = x.view(B, self.out_vectors, self.outdim)
        return out


class MLPRegressor(nn.Module):
    def __init__(self,
                 input_size,
                 out_vectors,
                 outdim_trasl,
                 hidden_sizes,
                 outdim_orient=3,
                 weight_orient=1.,
                 confidence_scores=False):
        """
          input_size: int
          out_vectors: int, num of output points
          hidden_sizes: list of int
            
          outdim: int, channel dims for each output point
          outdim_orient: orient channel dims for each output point
        """
        super(MLPRegressor, self).__init__()
        
        self.out_vectors = out_vectors
        self.outdim_trasl = outdim_trasl
        self.outdim_orient = outdim_orient
        self.weight_orient = weight_orient
        self.confidence_scores = confidence_scores

        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.linears.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_sizes[0]))
        
        # Add additional hidden layers if specified
        previous_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            self.linears.append(torch.nn.Linear(previous_size, size))
            self.batch_norms.append(torch.nn.BatchNorm1d(size))
            previous_size = size
        
        self.output_trasl = torch.nn.Linear(previous_size, out_vectors*outdim_trasl)

        if self.outdim_orient > 0:
            self.output_normals = torch.nn.Linear(previous_size, out_vectors*outdim_orient)
            self.tanh = nn.Tanh()

        if self.confidence_scores:
            self.out_confidence = torch.nn.Linear(previous_size, out_vectors)
  
    def forward(self, x, relative_pred=False):
        B = x.shape[0]

        if relative_pred:
            centroids_3dbboxes = x[:, :3].clone()  # (B, 3), centroid of each bounding box

        for linear, batch_norm in zip(self.linears, self.batch_norms):
            x = linear(x)
            x = batch_norm(x)
            x = F.relu(x)

        x_trasl = self.output_trasl(x)

        if self.outdim_orient > 0:
            x_normals = self.tanh(self.output_normals(x))
            x_normals = x_normals.view(B, -1, 3)
            x_normals = F.normalize(x_normals, dim=-1)
            x_normals *= self.weight_orient

            x_trasl = x_trasl.view(B, -1, 3)
            if relative_pred:
                # Predictions are relative to centroid of input 3D bounding boxes
                x_trasl += centroids_3dbboxes[:, None, :]  # add stroke dimension            

            out = torch.cat((x_trasl, x_normals), dim=-1)
            out = out.view(B, self.out_vectors, -1)  # (B, N, outdim_trasl+outdim_orient)
        else:
            out = x_trasl.view(B, self.out_vectors, self.outdim_trasl)

        if self.confidence_scores:
            scores = self.out_confidence(x)
            scores = scores.view(B, self.out_vectors, 1)  # (B, N, 1)
            return out, scores
        else:
            return out
    