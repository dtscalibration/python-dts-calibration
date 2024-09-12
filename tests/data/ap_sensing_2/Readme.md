# Data documentation - ap_sensing_2

## Exported Files:
For each measurment both file exports of the AP sensing N4386B were used:
	- .tra files 
		- more metadata available, like all device settings of configurations
		- PT100 Sensor data included
		- Signals of Distance, Temperature, Log-ratio and Attenuation
		- not included signals: Stokes and Anti-Stokes intensities
	- .xml files (using POSC export)
		- sparse metadate e.g. no device settings used in configuration
		- no PT100 Sensor data
		- Stokes and Anti-Stokes intensities included!!

## Measurement Setup:
- Duplex OM4 glass fibre connected to CH1 and CH2
- Fibre length: 98 m
- Splice at 98 m
- Two controlled sections:
	- hot bath 
		- Temp: ~50 °C 
		- Distance: 40 m - 52 m
		- Reference Temperature Sensor 2
	- empty cold bath
		- Temp environmal
		- Distance: 60 m - 72 m
		- Reference Temperature Sensor 1
	- Comments:
		- approx. 15m of fibre and a reference PT100 are mounted into a 3D-printed hollow torus
			-> Span is set to 12 m as DTS data shows artifacts near large temperature gradients
		- torus is submerged and fixed under water, small holes in torus make sure that it fills completely with water
		- water temperature outside torus is controlled
- heated Helix near splice
	- heated length = 2547 mm
	- distance helix to splice ~ 10-15 cm

## DTS measurement settings: (given in .seq file, which can by imported by AP sensing software - DTS Configurator)
- Sequence containing three Configurations
	- C1: Single ended from Channel 1
	- C2: Single ended from Channel 2
	- C3: Double ended from Channel 1