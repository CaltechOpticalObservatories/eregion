from typing import Any


_DEIMOS_FPTYPE_EXTRA_PRIKWS: dict[str, list[str]] = {
    "SCI" : ["DTU_POS", "CCD_SERN", "CCD_MPN"],
    "FCS" : []
}

_DEIMOS_IMTYPE_EXTRA_PRIKWS : dict[str, list[str]] = {
    "TDIWithIPhi" : ["TDI_TIME", "IPHI", "LED"],
    "TDIBiasWithIPhi" : ["TDI_TIME", "IPHI"]
}


def DEIMOS_imtype_header_identify(headers) -> dict[str, Any]:
    """Custom identifier task for DEIMOS headers"""


    #NOTE: current implementation in imagegen.py enforces primary HDU to be the first header.
    # But this seems a little fragile. Might be better to pass astropy HDUList into this function
    # rather than just dicts. That way we could use their API to identify e.g. primary HDU etc

    try:
        if headers[0]["INSTR"] != "DEIMOS_DTU":
            raise ValueError("this function intended only to work with DEIMOS DTU testdata")
    except KeyError as err:
        raise ValueError("INSTR keyword not found in what we thought was the primary HDU") from err


    #NOTE: obstime and seqnum should NOT be part of a thing called "image Type"
    # rather these should be extracted by a different extraction function into the top level
    #metadata of DetImage. But the current eregion design has it this way so I follow for now
    # in the interests of not changing too many things at once
    imtype = {"type" : headers[0]["IMAGE_TYPE"],
              "obstime" : headers[0]["TIMEUTC"],
              "seqnum" : headers[0]["SEQNUM"],
              "fptype" : headers[0]["DTU_FPTYPE"]}

    extra_kws = []

    if imtype["fptype"] not in _DEIMOS_FPTYPE_EXTRA_PRIKWS:
        raise ValueError(f"unrecognised focal plane type {imtype[fptype]}")

    extra_kws += _DEIMOS_FPTYPE_EXTRA_PRIKWS[imtype["fptype"]]

    if imtype["type"] not in _DEIMOS_IMTYPE_EXTRA_PRIKWS:
        raise ValueError(f"unrecognised image type {imtype[type]}")

    extra_kws += _DEIMOS_IMTYPE_EXTRA_PRIKWS[imtype["type"]]
    imtype |= {k : headers[0].get(k, None) for k in extra_kws if k in headers[0]}
    return imtype
    

    
    
