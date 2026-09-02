import inspect
from typing import Optional, Iterable, Container, Any, Type

def pack_argument_helper(exclude_args: Optional[Container[str] | Iterable[str]] = None, selfarg: Optional[Type] = None) -> dict[str, Any]:
    """ Gets all the arguments passed to the function which calls this function, and their current values, and packs them into a dictionary suitable for use as kwargs in some other call.

    parameters
    ----------

    exclude_args: Optional[Container[str] | Iterable[str]]
       some collection of strings which supports the "in" operator. The arguments named in this collection will not be included in the returned kwargs dict

    selfarg: Optional[Type]
       if this is called from inside an instance method or class method, supply the `self` or `cls` here, in order to remove it from the returned dict


    returns
    -------

    dict[str, Any]

        dictionary of arguments, which will include all the keyword and 
    

    """
    outerframe = inspect.currentframe().f_back
    args, vargs, kwargs, lcls = inspect.getargvalues(outerframe)
    outkwargs = {k : lcls[k] for k in args}
    if kwargs is not None:
        outkwargs.update(lcls[kwargs])

    if vargs is not None:
        raise TypeError("pack_argument_helper does not work with unnamed positional arguments")

    if selfarg is not None:
        selfargk = next( k for k,v in outkwargs.items() if v is selfarg)
        outkwargs.pop(selfargk)

    if exclude_args is not None:
        for exarg in exclude_args:
            if exarg in outkwargs:
                outkwargs.pop(exarg)

    return outkwargs

