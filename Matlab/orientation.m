function [ O ] = orientation(u, v, w)

O = det([ 1, u(1), u(2); 
          1, v(1), v(2); 
          1, w(1), w(2)]);

end