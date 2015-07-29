// Parameters
c_pt = 0.001;
r_pt = 0.5;
c_width  = 0.02;
c_height = 0.02;
c_length = 0.8;
r_radius = 1.0;


// Points
// channel and gate
Point(1)  = { c_length/2, c_width/2, c_height/2,c_pt};
Point(2)  = { c_length/6, c_width/2, c_height/2,c_pt};
Point(3)  = {          0, c_width/2, c_height/2,c_pt};
Point(4)  = {-c_length/6, c_width/2, c_height/2,c_pt};
Point(5)  = {-c_length/2, c_width/2, c_height/2,c_pt};
Point(6)  = { c_length/2,-c_width/2, c_height/2,c_pt};
Point(7)  = { c_length/6,-c_width/2, c_height/2,c_pt};
Point(8)  = {          0,-c_width/2, c_height/2,c_pt};
Point(9)  = {-c_length/6,-c_width/2, c_height/2,c_pt};
Point(10) = {-c_length/2,-c_width/2, c_height/2,c_pt};
Point(11) = { c_length/2, c_width/2,-c_height/2,c_pt};
Point(12) = { c_length/6, c_width/2,-c_height/2,c_pt};
Point(13) = {          0, c_width/2,-c_height/2,c_pt};
Point(14) = {-c_length/6, c_width/2,-c_height/2,c_pt};
Point(15) = {-c_length/2, c_width/2,-c_height/2,c_pt};
Point(16) = { c_length/2,-c_width/2,-c_height/2,c_pt};
Point(17) = { c_length/6,-c_width/2,-c_height/2,c_pt};
Point(18) = {          0,-c_width/2,-c_height/2,c_pt};
Point(19) = {-c_length/6,-c_width/2,-c_height/2,c_pt};
Point(20) = {-c_length/2,-c_width/2,-c_height/2,c_pt};
Point(21) = { c_length/2, r_radius, 0,r_pt};
Point(22) = { c_length/2,-r_radius, 0,r_pt};
Point(23) = { c_length/2, 0, r_radius,r_pt};
Point(24) = { c_length/2, 0,-r_radius,r_pt};
Point(25) = { c_length/2+r_radius, 0, 0,r_pt};
Point(26) = {-c_length/2, r_radius, 0,r_pt};
Point(27) = {-c_length/2,-r_radius, 0,r_pt};
Point(28) = {-c_length/2, 0, r_radius,r_pt};
Point(29) = {-c_length/2, 0,-r_radius,r_pt};
Point(30) = {-c_length/2-r_radius, 0, 0,r_pt};
Point(31) = { c_length/2, 0, 0,c_pt};
Point(32) = {-c_length/2, 0, 0,c_pt};

// Lines
Line(1) = {5, 15};
Line(2) = {15, 20};
Line(3) = {20, 10};
Line(4) = {10, 5};
Line(5) = {5, 4};
Line(6) = {4, 3};
Line(7) = {3, 2};
Line(8) = {2, 1};
Line(9) = {1, 6};
Line(10) = {6, 16};
Line(11) = {16, 11};
Line(12) = {11, 1};
Line(13) = {11, 12};
Line(14) = {12, 13};
Line(15) = {13, 14};
Line(16) = {14, 15};
Line(17) = {20, 19};
Line(18) = {19, 18};
Line(19) = {18, 17};
Line(20) = {17, 16};
Line(21) = {10, 9};
Line(22) = {9, 8};
Line(23) = {8, 7};
Line(24) = {7, 6};
Line(25) = {7, 17};
Line(26) = {17, 12};
Line(27) = {12, 2};
Line(28) = {2, 7};
Line(29) = {8, 18};
Line(30) = {18, 13};
Line(31) = {13, 3};
Line(32) = {3, 8};
Line(33) = {9, 19};
Line(34) = {19, 14};
Line(35) = {14, 4};
Line(36) = {4, 9};
Circle(37) = {29, 32, 26};
Circle(38) = {26, 32, 28};
Circle(39) = {28, 32, 27};
Circle(40) = {27, 32, 29};
Circle(41) = {29, 32, 30};
Circle(42) = {30, 32, 26};
Circle(43) = {28, 32, 30};
Circle(44) = {30, 32, 27};
Circle(45) = {22, 31, 23};
Circle(46) = {23, 31, 21};
Circle(47) = {21, 31, 24};
Circle(48) = {24, 31, 22};
Circle(49) = {22, 31, 25};
Circle(50) = {25, 31, 23};
Circle(51) = {24, 31, 25};
Circle(52) = {25, 31, 21};

// Surfaces

Line Loop(53) = {16, -1, 5, -35};
Plane Surface(54) = {53};
Line Loop(55) = {15, 35, 6, -31};
Plane Surface(56) = {55};
Line Loop(57) = {31, 7, -27, 14};
Plane Surface(58) = {57};
Line Loop(59) = {27, 8, -12, 13};
Plane Surface(60) = {59};
Line Loop(61) = {3, 21, 33, -17};
Plane Surface(62) = {61};
Line Loop(63) = {22, 29, -18, -33};
Plane Surface(64) = {63};
Line Loop(65) = {23, 25, -19, -29};
Plane Surface(66) = {65};
Line Loop(67) = {25, 20, -10, -24};
Plane Surface(68) = {67};
Line Loop(69) = {20, 11, 13, -26};
Plane Surface(70) = {69};
Line Loop(71) = {26, 14, -30, 19};
Plane Surface(72) = {71};
Line Loop(73) = {30, 15, -34, 18};
Plane Surface(74) = {73};
Line Loop(75) = {34, 16, 2, 17};
Plane Surface(76) = {75};
Line Loop(77) = {5, 36, -21, 4};
Plane Surface(78) = {77};
Line Loop(79) = {36, 22, -32, -6};
Plane Surface(80) = {79};
Line Loop(81) = {7, 28, -23, -32};
Plane Surface(82) = {81};
Line Loop(83) = {28, 24, -9, -8};
Plane Surface(84) = {83};
Line Loop(85) = {46, 47, 48, 45};
Line Loop(86) = {12, 9, 10, 11};
Plane Surface(87) = {85, 86};
Line Loop(88) = {37, 38, 39, 40};
Line Loop(89) = {4, 1, 2, 3};
Plane Surface(90) = {88, 89};
Line Loop(91) = {42, 38, 43};
Ruled Surface(92) = {91};
Line Loop(93) = {43, 44, -39};
Ruled Surface(94) = {93};
Line Loop(95) = {40, 41, 44};
Ruled Surface(96) = {95};
Line Loop(97) = {41, 42, -37};
Ruled Surface(98) = {97};
Line Loop(99) = {52, 47, 51};
Ruled Surface(100) = {99};
Line Loop(101) = {51, -49, -48};
Ruled Surface(102) = {101};
Line Loop(103) = {50, -45, 49};
Ruled Surface(104) = {103};
Line Loop(105) = {50, 46, -52};
Ruled Surface(106) = {105};

// Volume
Surface Loop(107) = {96, 90, 98, 92, 94, 62, 78, 54, 76, 74, 72, 70, 68, 66, 82, 58, 56, 80, 64, 60, 84, 87, 106, 104, 102, 100};
Volume(108) = {107};

// Physical Entities
Physical Surface(1) = {98, 92, 96, 94};
Physical Surface(2) = {106, 100, 102, 104};
Physical Surface(3) = {56, 58, 66, 64};
Physical Surface(4) = {54, 62, 68, 60};
Physical Surface(5) = {90, 87};

Physical Volume(1) = {108};
