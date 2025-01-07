clear
format long


site='HCKM';
fileMES1='01-Nov-2009_01-Aug-2023_HCKM_daily.mat';
dateS=datetime(2009,12,01);
dateE=datetime(2023,05,31);
%site='FARM';
%fileMES1='01-Jan-2008_01-Aug-2023_FARM_daily.mat';
%dateS=datetime(2008,03,01);
%dateE=datetime(2023,05,31);

%load Mesonet, Climate Data and Thresholds
dircli1='/Users/erappin/Documents/Mesonet/ClimateIndices/sitesTEST_CCindices/';
dircli=strcat(strcat(dircli1,site,'/'));
filethresh=strcat(dircli,strcat(site,"_CLIthresh_daily"));
load(filethresh);
filemes=strcat(dircli,fileMES1);
load(filemes);

TIME_full=TT_dailyMES.TimestampCollected;
sTIME_full=TIME_full(1);
eTIME_full=TIME_full(end);
[sYf,sMf,sDf]=ymd(TIME_full(1));
[eYf,eMf,eDf]=ymd(TIME_full(end));
isD=find(all(ismember(TIME_full,dateS),2));
ieD=find(all(ismember(TIME_full,dateE),2));

VARold=["TAIR" "TAIRx" "TAIRn" "DWPT" "PRCP" "PRES" "RELH" "SM02" "SRAD" "WDIR" "WSPD" "WSMX"];
VAR=["TAIR_season" "TAIRx_season" "TAIRn_season" "DWPT_season" "PRCP_season" "PRES_season" ... 
     "RELH_season" "SM02_season" "SRAD_season" "WDIR_season" "WSPD_season" "WSMX_season"]';
TT=TT_dailyMES(isD:ieD,:);
TT=removevars(TT,{'SM20' 'SM04' 'SM40' 'SM08' 'ST02' 'ST20' 'ST04' 'ST40' 'ST08' 'WSMN'});
TT=renamevars(TT,VARold,VAR);
TIME=TT.TimestampCollected;

YEAR=unique(TIME.Year);
DATAseason.year=YEAR;
SEAS=["Winter" "Spring" "Summer" "Fall"]';
DATAseason.season=SEAS;
DATAseason.var=VAR;

springMonth = month(TIME) == 3 | month(TIME) == 4 | month(TIME) == 5;
summerMonth = month(TIME) == 6 | month(TIME) == 7 | month(TIME) == 8;
fallMonth   = month(TIME) == 9 | month(TIME) == 10 | month(TIME) == 11;
winterMonth = month(TIME) == 1 | month(TIME) == 2 | month(TIME) == 12;
% Create grouping variable for winter year
% (treating consecutive Dec, Jan Feb as winter that year)
% Extract winter year as the year corresponding to January
winterYr = year(TIME(winterMonth));
winterYr(month(TIME(winterMonth))==12) = winterYr(month(TIME(winterMonth))==12)+1;
winter = datetime(TIME(winterMonth));
TTwinter=TT(winter,:);
springYr = year(TIME(springMonth));
spring = datetime(TIME(springMonth));
TTspring=TT(spring,:);
summerYr = year(TIME(summerMonth));
summer = datetime(TIME(summerMonth));
TTsummer=TT(summer,:);
fallYr = year(TIME(fallMonth));
fall = datetime(TIME(fallMonth));
TTfall=TT(fall,:);

ny=numel(YEAR);
ns=numel(SEAS);
nv=numel(VAR);

DATAwinter=cell(ny,1);
DATAspring=cell(ny,1);
DATAsummer=cell(ny,1);
DATAfall=cell(ny,1);

[ys,ms,ds]=ymd(TIME(1));
[ye,me,de]=ymd(TIME(end));

%winter start
if ms == 12
   for k=1:ny-1
       
       TTwinterA=TTwinter;
       lower=datetime(YEAR(k),12,01);
       upper=datetime(YEAR(k+1),02,29);
       winterY=isbetween(winter,lower,upper);
       TTwinterA(~winterY,:)=[];
       DATAwinter{k,1}=TTwinterA;  
       isEM=isempty(DATAwinter{k,1});
       if isEM==1
          DATAwinter{k,1}=[];
       end
      
       TTspringA=TTspring;
       lower=datetime(YEAR(k+1),02,29);
       upper=datetime(YEAR(k+1),05,31);
       springY=isbetween(spring,lower,upper);
       TTspringA(~springY,:)=[];
       DATAspring{k+1,1}=TTspringA;
       isEM=isempty(DATAspring{k+1,1});
       if isEM==1
          DATAspring{k+1,1}=[];
       end
      
       TTsummerA=TTsummer;
       lower=datetime(YEAR(k+1),06,01);
       upper=datetime(YEAR(k+1),08,31);
       summerY=isbetween(summer,lower,upper);
       TTsummerA(~summerY,:)=[];
       DATAsummer{k+1,1}=TTsummerA;
       isEM=isempty(DATAsummer{k+1,1});
       if isEM==1
          DATAsummer{k+1,1}=[];
       end

       TTfallA=TTfall;
       lower=datetime(YEAR(k+1),09,01);
       upper=datetime(YEAR(k+1),11,30);
       fallY=isbetween(fall,lower,upper);
       TTfallA(~fallY,:)=[];
       DATAfall{k+1,1}=TTfallA;
       isEM=isempty(DATAfall{k+1,1});
       if isEM==1
          DATAfall{k+1,1}=[];
       end

   end
else
    for k=1:ny-1
       TTwinterA=TTwinter;
       lower=datetime(YEAR(k),12,01);
       upper=datetime(YEAR(k+1),02,29);
       winterY=isbetween(winter,lower,upper);
       TTwinterA(~winterY,:)=[];
       DATAwinter{k+1,1}=TTwinterA;  
       isEM=isempty(DATAwinter{k+1,1});
       if isEM==1
          DATAwinter{k+1,1}=[];
       end
    end

    for k=1:ny
       TTspringA=TTspring;
       lower=datetime(YEAR(k),02,29);
       upper=datetime(YEAR(k),05,31);
       springY=isbetween(spring,lower,upper);
       TTspringA(~springY,:)=[];
       DATAspring{k,1}=TTspringA;
       isEM=isempty(DATAspring{k,1});
       if isEM==1
          DATAspring{k,1}=[];
       end
      
       TTsummerA=TTsummer;
       lower=datetime(YEAR(k),06,01);
       upper=datetime(YEAR(k),08,31);
       summerY=isbetween(summer,lower,upper);
       TTsummerA(~summerY,:)=[];
       DATAsummer{k,1}=TTsummerA;
       isEM=isempty(DATAsummer{k,1});
       if isEM==1
          DATAsummer{k,1}=[];
       end

       TTfallA=TTfall;
       lower=datetime(YEAR(k),09,01);
       upper=datetime(YEAR(k),11,30);
       fallY=isbetween(fall,lower,upper);
       TTfallA(~fallY,:)=[];
       DATAfall{k,1}=TTfallA;
       isEM=isempty(DATAfall{k,1});
       if isEM==1
          DATAfall{k,1}=[];
       end
    end
end

DATAseasons{1,:}=DATAwinter; 
DATAseasons{2,:}=DATAspring; 
DATAseasons{3,:}=DATAsummer;
DATAseasons{4,:}=DATAfall;
DATAseason.data=DATAseasons;





