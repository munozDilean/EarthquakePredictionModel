# Earthquake Prediction Model

Final project for the Deep Learning course by Wright State

- Participants: Dilean Munoz, Jinho Nam, David Tincher

This codebase was developed using python 3.14

## Running the model

1. Set up a virtual Environment

```zsh
python3 -m venv .venv
```

2. Activate your virtual environment and select it in vs code. **This will depend on what OS you are running the code in**

3. Install all dependencies inside your .venv

```zsh
pip3 install -r requirements.txt
```

4. Run the program inside the venv

```zsh
python3 main.py
```

## About Dataset

[Datasets](https://www.kaggle.com/datasets/warcoder/earthquake-dataset?resource=download) contain records of 782 earthquakes from 1/1/2001 to 1/1/2023. The meaning of all columns is as follows:

- title: title name given to the earthquake
- magnitude: The magnitude of the earthquake
- date_time: date and time
- cdi: The maximum reported intensity for the event range
- mmi: The maximum estimated instrumental intensity for the event
- alert: The alert level - “green”, “yellow”, “orange”, and “red”
- tsunami: "1" for events in oceanic regions and "0" otherwise
- sig: A number describing how significant the event is. Larger numbers indicate a more significant event. This value is determined on a number of factors, including: magnitude, maximum MMI, felt reports, and estimated impact
- net: The ID of a data contributor. Identifies the network considered to be the preferred source of information for this event.
- nst: The total number of seismic stations used to determine earthquake location.
- dmin: Horizontal distance from the epicenter to the nearest station
- gap: The largest azimuthal gap between azimuthally adjacent stations (in degrees). In general, the smaller this number, the more reliable is the calculated horizontal position of the earthquake. Earthquake locations in which the azimuthal gap exceeds 180 degrees typically have large location and depth uncertainties
- magType: The method or algorithm used to calculate the preferred magnitude for the event
- depth: The depth where the earthquake begins to rupture
- latitude / longitude: coordinate system by means of which the position or location of any place on Earth's surface can be determined and described
- location: location within the country
- continent: continent of the earthquake hit country
- country: affected country

## TODO

- [x] Download data to repo
- [ ] Setup Dataset for model
  - [ ] Training Data
  - [ ] Validation Data
  - [ ] Test Data
- [ ] Setup Training loop
