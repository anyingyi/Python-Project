## JSON format of order files

| key | key | key | value format | description |
| :-----| :----| :----|:---- |:---- |
| estimateCode |  |  | String | Identification of the input files. |
| algorithmBaseParamDto | | | Object| Basic data |
| | platformDtoList | | Array | Platform info. |
| | | platformCode | String | Identification of the platform. |
| | | isMustFirst | Boolean | Whether the platform  should be first visited (bonded warehouse). |
| | truckTypeDtoList | | Array | Truck info. |
| | | truckTypeId | String | Identification of the truck. |
| | | truckTypeCode | String | Unique truck code. |
| | | truckTypeName | String | Unique truck name. |
| | | length | Float | Truck length (mm). |
| | | width | Float | Truck width (mm). |
| | | height | Float | Truck height (mm). |
| | | maxLoad | Float | Carrying capacity of the truck (kg) |
| | truckTypeMap |  | Object | Map format of truckTypeDtoList, key is truckTypeId |
| | distanceMap	 |  | Object | Key is two platform codes connected by “+”: e.g. “platform01+platform02”; Value is the float value of the distance (m) between them. |
| boxes	|  |  | Array | Boxes info. |
| | spuBoxId |  | String | Identification of the box. |
| | platformCode |  | String | Code of the platform the box is belonging to. |
| | length | | Float | Box length (mm). |
| | width |  | Float | Box width (mm). |
| | height |  | Float | Box height (mm). |
| | weight |  | Float | Box weight (kg). |


## 
