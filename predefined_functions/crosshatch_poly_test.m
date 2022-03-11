% 
% Some simple examples of crosshatch_poly in use
%

clear;

x = 2 + [0 3 4 0];
y = 1 + [0 0 3 4];

%%%% SIMPLE EXAMPLE
figure(1)
[h,g] = crosshatch_poly(x, y, 30, 0.25, ...
			'edgestyle', '-', 'edgecolor', 'b', 'edgewidth', 2, ...
			'linestyle', '--', 'linecolor', 'r', 'linewidth', 1);
set(gca, 'xlim', [0 7]);
set(gca, 'ylim', [0 7]);
axis square



%%%% CROSS HATCHING EXAMPLE
figure(2)
crosshatch_poly(x, y, 30, 0.25, ...
		'edgestyle', '-', 'edgecolor', 'k', 'edgewidth', 1, ...
		'linestyle', '-', 'linecolor', 'r', 'linewidth', 1, ...
		'backgroundcolor', [0.7 0.95 1]);
crosshatch_poly(x, y, 120, 0.25, ...
		'edgewidth', 0, ...
		'linestyle', '-', 'linecolor', 'r', 'linewidth', 1, ...
		'hold', 1);
set(gca, 'xlim', [0 7]);
set(gca, 'ylim', [0 7]);
axis square

%%%% OVERLAPPING HATCHING EXAMPLE
N = 9;
theta = 0:2*pi/N:2*pi;
x1 = 3 + cos(theta);
y1 = 3 + sin(theta);

x2 = [3 5.6 5.6 3];
y2 = [2.4 2.4 2.8 2.8];

N2 = 30;
theta2 = 0:2*pi/N2:2*pi;
r = 0.6;
x3 = 4 + r*cos(theta2);
y3 = 3.5 + r*sin(theta2);

r = 0.4;
x4 = 2.3 + r*cos(theta2);
y4 = 2.3 + r*sin(theta2);

x5 = [2 4 4.5 3 2];
y5 = [4.4 4.2 5 5.5 5];

figure(3)
crosshatch_poly(x1, y1, 30, 0.25, ...
		'linestyle', '-', 'linecolor', 'k', 'linewidth', 1);
crosshatch_poly(x2, y2, 0, 0.2, 'edgecolor', 'b', ...
		'linestyle', '-', 'linecolor', 'b', 'linewidth', 1, 'hold', 1);
crosshatch_poly(x3, y3, 60, 0.2, 'edgecolor', 'g', ...
		'linestyle', '-', 'linecolor', 'g', 'linewidth', 1, 'hold', 1);
crosshatch_poly(x3, y3, 150, 0.2, 'edgecolor', 'g', ...
		'linestyle', '-', 'linecolor', 'g', 'linewidth', 1, 'hold', 1);
crosshatch_poly(x4, y4, 150, 0.15, 'edgecolor', 'm', ...
		'linestyle', '-', 'linecolor', 'm', 'linewidth', 1, 'hold', 1, ...
		'backgroundcolor', [1 1 1]);
crosshatch_poly(x5, y5, 90, 0.1, 'edgecolor', 'k', ...
		'linestyle', ':', 'linecolor', 'k', 'linewidth', 1, 'hold', 1, ...
		'backgroundcolor', [1 0.9 1]);
set(gca, 'xlim', [1.5 5.7]);
set(gca, 'ylim', [1.5 5.7]);
axis square
axis off

set(gcf, 'PaperPosition', [0 0 2 2]);
print('-dpng', 'cross_hatch_poly.png');
print('-depsc', 'cross_hatch_poly.eps');

% Current weirdness is that in the Matlab plot window, or EPS we can overlay one background on
% another, but not in the PNG where background fills are transparent.



