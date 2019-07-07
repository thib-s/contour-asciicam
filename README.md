# contour-asciicam

```                                                                                
             __unm2   _____ e                                                   
      _unm1 _qr _un1       5q 1e                                                
 e8888oooe__8m1              0s  2_                                             
 (  a888nnu())88uu_            0*  0                                    _e5i )  
 3-qu          0mu(588u_                                          _s88h   o4l0l 
  8q8               dr  n88888888((((((m888888uuuu__       _uo8m1     _2    8 8 
   u8                q_       #(            )2mnuu_  28888(      _um        0_) 
   )8_               0l#   e(q8                     o    il _un1                
   80)               2q8    8_)(r                    l  ui1                     
    k0l              (8i     k28qr                   )qi                        
     es4            808      0xhk(                   )#                         
       -q88u_      e0y        5(0#                   (8                         
           5n88())2*           d0uk                 qi)                         
                                 om8_               (e                          
                                   0u((08nug--enma(1d                           
                                          )288521                               
```

yet an other image to assci art converter, however this one extract the contours of the image


what to install
---------------

install opencv3 and numpy. (some conda env is coming soon !)

usage
-----

```
$ python asciimatcher.py --help
usage: python asciimatcher.py path

convert image to ascii art by detecting it's contours

positional arguments:
  path                  location of the image to convert

optional arguments:
  -h, --help            show this help message and exit
  --outwidth OUTWIDTH   width of the output image (number of chars)
  --charset CHARSET     list of allowed characters for the output string
  --canny_alpha CANNY_ALPHA
                        values between 0 and 1, the higher it is the more
                        details there is
  --duplicate_spaces DUPLICATE_SPACES
                        repeat spaces as some devices shink those
  --display_canny       if set, display the canny filter results for debugging

```