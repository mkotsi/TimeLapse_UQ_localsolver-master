%clear

% size of domain
sX = [25 44];
% number of pixels to taper over (x,y)
tap_width = [10 18];

taperX =exptap(1,sX(1),tap_width(1),[1:sX(1)]);
taperY =exptap(1,sX(2),tap_width(2),[1:sX(2)]);

taper_2d=repmat(taperX',[1 sX(2)]).*repmat(taperY,[sX(1) 1]);

subplot(1,2,1)
plot(taperX)
hold on
plot(taperY)
hold off
title('Tapers in XY direction')

subplot(1,2,2)
imagesc(taper_2d)
colorbar
title('2D taper mask')
axis equal tight

%%
function [tap]=exptap(ts1x,te2x,tlen,x)
%function [tap]=exptap(ts1x,te2x,tlen,x)
%ts1x is the start of the taper on the left
%te2x is the end of the taper on the right
%tlen is the number of points over which to taper
%x is the ordinates for the taper
  
  tap=ones(size(x));
  
  te1x=ts1x+tlen;
  ts2x=te2x-tlen;
  
  tap(ts1x+1:te1x-1)=exp(-(x(te1x)-x(ts1x))./(x(ts1x+1:te1x-1)-x(ts1x)).* ...
			exp(-(x(ts1x)-x(te1x))./(x(ts1x+1:te1x-1)-x(te1x))));
  %ramp down
  tap(ts2x+1:te2x-1)=exp(-(x(te2x)-x(ts2x))./(-x(ts2x+1:te2x-1)+x(te2x)).* ...
			exp(-(x(ts2x)-x(te2x))./(-x(ts2x+1:te2x-1)+ ...
						 x(ts2x))));
  tap(te2x:length(x))=0;
tap(1:ts1x)=0;

end