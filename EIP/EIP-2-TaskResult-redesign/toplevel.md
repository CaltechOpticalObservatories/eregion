EIP-2 redesign the TaskResult
-----------------------------

## Goals

	- Make TaskResults more generic and composable (so that each type of Task no longer needs to always define custom names etc in the Result)
	- (relatedly) make TaskResult more ergonomic to use with Task (try to reduce the number of `.data` type calls needed in passing around results
	- Through the use of pydantic custom serialization, allow uniform serializing / de-serializing to FITS and (in future) other binary container formats with decent round-tripping, generic to TaskResult (i.e. only minimal customization points needed for special TaskResults e.g. including diffims)
	- Make the smallest TaskResult[T] generic (e.g. a single table row, a single diffim etc), so that a generic collection of these results can be a Mappable keyed on a named-tuple like type, and automatically converted to a tabular like results type if needed
	- Allow for intelligent disk backing and checkpointing of TaskResults in an opt-in fashion, e.g. to enable easier resumability of manually coded pipelines
	- Allow for automatic conversion to and from pandas DataFrames
	
	
##  Non-goals / open questions for this EIP
    - Not aiming to make all tasks pass and return the same TaskResult class (the point is to use pydantic data validation on it)
	- Not aiming to hard specify a particular FITS / netCDF metadata schema (though a default one will naturally arise)

