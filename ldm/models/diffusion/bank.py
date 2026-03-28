from .misc_4ddpm import *


from ldm.modules.attention import BasicTransformerBlock
class Bank:
    def __init__(self,reader:nn.Module, writer:nn.Module) -> None:
        """
        For the DFS model, mark every BasicTransformerBlock with name_4bank and isReader_4bank flags.
        Similar logic applies for the writer while checking for BasicTransformerBlock instances.
        """
        self.name2data = {}
        self.name2count = {}  # track how many times each name has been retrieved
        self.WHEN_clear_a_field = 2  # clear the entry after this many gets
        skip_names = [
            'input_blocks.1.1.transformer_blocks.0',
            'input_blocks.2.1.transformer_blocks.0',
            # 'input_blocks.4.1.transformer_blocks.0',
            # 'input_blocks.5.1.transformer_blocks.0',
            # 'input_blocks.7.1.transformer_blocks.0',
            # 'input_blocks.8.1.transformer_blocks.0',
            ##-----------all middle and output_blocks (everything outside input_blocks)----
            'middle_block.1.transformer_blocks.0',
            'output_blocks.3.1.transformer_blocks.0',
            'output_blocks.4.1.transformer_blocks.0',
            'output_blocks.5.1.transformer_blocks.0',
            'output_blocks.6.1.transformer_blocks.0',
            'output_blocks.7.1.transformer_blocks.0',
            'output_blocks.8.1.transformer_blocks.0',
            'output_blocks.9.1.transformer_blocks.0',
            'output_blocks.10.1.transformer_blocks.0',
            'output_blocks.11.1.transformer_blocks.0',
        ]
        # print(f"{skip_names=}")
        
        l_name = []
        for name, _module in writer.named_modules():
            if isinstance(_module, BasicTransformerBlock):
                if DEBUG:
                    print(f"{name=}")
                if name in skip_names:
                    # print(f"skip {name=}")
                    continue
                _module.bank = self
                _module.name4bank = name
                _module.isReader_4bank = False
                l_name.append(name)
        # print(f"{l_name=}")
                
        for name, _module in reader.named_modules():
            if isinstance(_module, BasicTransformerBlock):
                if name not in l_name:
                    continue
                _module.bank = self
                _module.name4bank = name
                _module.isReader_4bank = True
    def set(self,name,data):
        self.name2data[name] = data
        # self.name2count[name] = 0
    def get(self,name):
        printC('bank get', name)
        if name in self.name2data:
            if name not in self.name2count:
                self.name2count[name] = 0
            self.name2count[name] += 1
            data = self.name2data[name]
            if self.name2count[name] >= self.WHEN_clear_a_field: # once the max get count is reached, remove the entry
                del self.name2data[name]
                del self.name2count[name]
            return data
        raise Exception(f"{name}\n{list(self.name2data.keys())}")
        return None
    def clear(self,):
        printC('clear')
        printC('mean ct:', sum( self.name2count.values() ) / len( self.name2count.values() ) if len( self.name2count.values() )>0 else 'null'   )
        self.name2data.clear()
        self.name2count.clear()
