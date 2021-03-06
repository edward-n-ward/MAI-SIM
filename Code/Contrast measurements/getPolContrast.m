% Simple program to estimate maximum modulation depth as a
% function of polarisation angle.
% Code adapted from:
%    Kevin O Holleran and Michael Shaw, "Polarization effects on contrast...
%    in structured illumination microscopy," Opt. Lett. 37, 4603-4605 (2012)



clear all; close all;
NA = 1.2;
n = 1.34;
res = 1.66;

theta_max = asin(NA/n);
res_fac = res-1;
sim_angle = atan(res_fac*tan(theta_max));
sim_angle = round(sim_angle,2);
theta = 0:0.01:pi/2;
phi = -pi/2:0.01:pi/2;
[thetas, phis] = meshgrid(theta,phi);
A = abs(cos(phis).*cos(phis).*cos(thetas) + sin(phis).*sin(phis));
B = abs(sin(thetas/2).*sin(thetas/2).*sin(2*phis));
C = abs(cos(phis).*sin(thetas));
Cont = abs(A.*A+B.*B-C.*C);
figure('Renderer', 'painters', 'Position', [100 10 1200 600]);
subplot(1,2,1);
surf(thetas(end-158:end,:),phis(end-158:end,:),Cont(end-158:end,:),'LineStyle','none');
xlabel('SIM incident angle (radians)');
ylabel('Polarisation angle (radians)');
zlabel('Modulation depth')
axis square

ind = round(sim_angle/0.01);
subplot(1,2,2);
Xs = linspace(-90,90,315);
plot(Xs,circshift(squeeze(Cont(:,ind)),round(length(Xs)/2)))
axis square
title("Modulation depth at: "+ num2str(180*sim_angle/pi) + " incident angle")
xlabel('Deviation from s polarisation (degrees)');
ylabel('Modulation depth');
axis([-90 90 0 1])

