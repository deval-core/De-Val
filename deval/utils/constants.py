from pydantic import BaseModel


class Constants(BaseModel):
    subnet_uid:int = 15
    name:str = "de_val"

    num_uids_total:int = 256
    max_model_size_gbs:int = 18 # allows for 8B models
    tier_improvement_threshold:float = 1.08
        


constants = Constants()