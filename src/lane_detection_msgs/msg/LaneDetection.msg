uint8 L3 = 0
uint8 L2 = 1
uint8 L1 = 2
uint8 R1 = 3
uint8 R2 = 4
uint8 R3 = 5

# coeffs for the line of best fit on xy-plane
# y(x) = p[0] * x^3 + p[1] * x^2 + p[2] * x^1 + p[3] * x^0 
float32[4] coeffs

# lane label
uint8 label

# The confidence score of the detected lane
float32 confidence


