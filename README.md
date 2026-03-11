# attune
Interactive meditation audio-visual art installation. The user sits in front of a candle, and is asked to focus their gaze upon the flame. There is a camera placed behind the candle, recording the user's face. track.py reads in the live camera data and uses a computer vision program to track the user's eyes. Based on how still the user's eyes stay over time, the program gives a 'focus score' from 0 to 1.

The focus score is then sent via OSC to a Max MSP patch that plays a continuous tone. When the focus score is low, the tone is interrupted by dissonance and chaos, showing the meditator that they have lost their concentration. While the focus score is high, the tone remains stable and harmonious. 
