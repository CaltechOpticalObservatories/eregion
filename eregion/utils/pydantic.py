from typing import Any, Type, Optional, TypeVar
from pydantic import BaseModel, create_model, Field
from copy import deepcopy
from datamodels import TaskResult


ModelT = TypeVar("ModelT", bound=BaseModel)

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
                        __base__=TaskResult)


    def add_result_item(self, single_result: ModelT) -> None:
        for fldname, fldval in single_result.items():
            if not hasattr(self, fldname):
                setattr(self, fldname, iterable_type())
            out = getattr(self, fldname)
            iterable_type_append_method(out, fldval)




    op_model.add_result_item = add_result_item
    return op_model
