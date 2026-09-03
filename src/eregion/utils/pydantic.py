from typing import Any, Type, Optional, TypeVar
from pydantic import BaseModel, create_model, Field
from copy import deepcopy
from eregion.datamodels import TaskResult
import pandas as pd
import os
from eregion.utils import save_dataframe_to_fits

ModelT = TypeVar("ModelT", bound=BaseModel)


class TabularTaskResult(TaskResult):
    SAVE_FILE_NAME: str = "result.fits"
    _ITERABLE_TYPE: Type = list
    _ITERABLE_TYPE_APPEND_METHOD: tuple[callable] = (list.append,)
    def add_result_item(self, single_result: ModelT) -> None:
        for fldname, fldval in single_result.items():
            if not hasattr(self, fldname):
                setattr(self, fldname, self._ITERABLE_TYPE())
            out = getattr(self, fldname)

            #NOTE: the tuple hack is to prevent this being set as a class method
            # on our class
            self._ITERABLE_TYPE_APPEND_METHOD[0](out, fldval)

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        df = dataframe_from_tabular_model(self)
        save_dataframe_to_fits(df, os.path.join(filepath, self.SAVE_FILE_NAME))
        super().save(filepath)




def generate_iterable_model(single_model: Type[ModelT], new_model_name: str,
                            index_field_name: Optional[str] = None,
                            index_field_type: Optional[Type] = None,
                            index_field_def: Optional[Field] = None,
                            iterable_type: Type = list,
                            iterable_type_append_method: callable = list.append,
                            **kwargs) -> Type[TaskResult]:

    out_fields: dict[str, tuple(Type, Field)] = dict()

    for infieldname, infieldprops in single_model.model_fields.items():
        #the type of the new field is just the selected iterable of the original field
        #the name is the same

        outfldprops = deepcopy(infieldprops)
        outfldprops.annotation = iterable_type[infieldprops.annotation]
        if hasattr(outfldprops, "default") and outfldprops.default is None:
            empty = iterable_type()
            outfldprops.default = deepcopy(empty)
        out_fields[infieldname] = (outfldprops.annotation, outfldprops)

    if index_field_name is not None:
        if index_field_def is None:
            raise ValueError("must supply the name of the index field")
        if index_field_type is None:
            raise ValueError("must supply the type of the index field")
        out_fields[index_field_name] = (index_field_type, index_field_def)

    op_model =  create_model(new_model_name, **out_fields, __config__=kwargs,
                             __base__=TabularTaskResult)


    op_model._ITERABLE_TYPE = iterable_type
    op_model._ITERABLE_TYPE_APPEND_METHOD = (iterable_type_append_method,)
    return op_model


def dataframe_from_tabular_model(model: TaskResult) -> pd.DataFrame:
    dump = model.model_dump()
    for k in model._metadata_field_names():
        del[dump[k]]
    return pd.DataFrame(data=dump)
