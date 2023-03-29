import torch 
import onnx 
import onnxruntime 
import numpy as np 
 
class DebugOp(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, x, name): 
        return x 
 
    @staticmethod 
    def symbolic(g, x, name): 
        return g.op("my::Debug", x, name_s=name) 
 
debug_apply = DebugOp.apply 
 
class Debugger(): 
    def __init__(self): 
        super().__init__() 
        self.torch_value = dict() 
        self.onnx_value = dict() 
        self.output_debug_name = [] 
 
    def debug(self, x, name): 
        self.torch_value[name] = x.detach().cpu().numpy() 
        return debug_apply(x, name) 
 
    def extract_debug_model(self, input_path, output_path): 
        model = onnx.load(input_path) 
        inputs = [input.name for input in model.graph.input] 
        outputs = [] 
 
        for node in model.graph.node: 
            if node.op_type == 'Debug': 
                debug_name = node.attribute[0].s.decode('ASCII') 
                self.output_debug_name.append(debug_name) 
 
                output_name = node.output[0] 
                outputs.append(output_name) 
 
                node.op_type = 'Identity' 
                node.domain = '' 
                del node.attribute[:] 
        e = onnx.utils.Extractor(model) 
        extracted = e.extract_model(inputs, outputs) 
        onnx.save(extracted, output_path) 
 
    def run_debug_model(self, input, debug_model): 
        sess = onnxruntime.InferenceSession(debug_model,  
        providers=['CPUExecutionProvider']) 
 
        onnx_outputs = sess.run(None, input) 
        for name, value in zip(self.output_debug_name, onnx_outputs): 
            self.onnx_value[name] = value 
 
    def print_debug_result(self): 
        for name in self.torch_value.keys(): 
            if name in self.onnx_value: 
                diff = np.abs(self.torch_value[name] - self.onnx_value[name])
                print(f"\n{name} shape:{diff.shape} with L1 diff {np.sum(diff)}, min diff:{np.min(diff)}, max diff:{np.max(diff)}")
                mse = np.mean(self.torch_value[name] - self.onnx_value[name])**2
                print(f"{name} MSE: {mse}") 