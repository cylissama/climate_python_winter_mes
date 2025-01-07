clear
format long

site='HCKM';
fileMES1='01-Nov-2009_01-Aug-2023_HCKM_daily.mat';
dateS=datetime(2009,11,01);
dateE=datetime(2023,07,31);

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
VAR=["TAIR_month" "TAIRx_month" "TAIRn_month" "DWPT_month" "PRCP_month" "PRES_month" ...
        "RELH_month" "SM02_month" "SRAD_month" "WDIR_month" "WSPD_month" "WSMX_month"];
TT=TT_dailyMES(isD:ieD,:);
TT=removevars(TT,{'SM20' 'SM04' 'SM40' 'SM08' 'ST02' 'ST20' 'ST04' 'ST40' 'ST08' 'WSMN'});
TT=renamevars(TT,VARold,VAR);
TIME=TT.TimestampCollected;

YEAR=unique(TIME.Year);
DATAmonth.year=YEAR;
MNTH=["Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul" "Aug" "Sep" "Oct" "Nov" "Dec"]';
DATAmonth.month=MNTH;
DATAmonth.var=VAR;

janYr=year(TIME(month(TIME) == 1));
january=datetime(TIME(month(TIME) == 1));
TTjan=TT(january,:);
febYr=year(TIME(month(TIME) == 2));
febuary=datetime(TIME(month(TIME) == 2));
TTfeb=TT(febuary,:);
marYr=year(TIME(month(TIME) == 3));
march=datetime(TIME(month(TIME) == 3));
TTmar=TT(march,:);
aprYr=year(TIME(month(TIME) == 4));
april=datetime(TIME(month(TIME) == 4));
TTapr=TT(april,:);
mayYr=year(TIME(month(TIME) == 5));
may=datetime(TIME(month(TIME) == 5));
TTmay=TT(may,:);
junYr=year(TIME(month(TIME) == 6));
june=datetime(TIME(month(TIME) == 6));
TTjun=TT(june,:);
julYr=year(TIME(month(TIME) == 7));
july=datetime(TIME(month(TIME) == 7));
TTjul=TT(july,:);
augYr=year(TIME(month(TIME) == 8));
august=datetime(TIME(month(TIME) == 8));
TTaug=TT(august,:);
sepYr=year(TIME(month(TIME) == 9));
september=datetime(TIME(month(TIME) == 9));
TTsep=TT(september,:);
octYr=year(TIME(month(TIME) == 10));
october=datetime(TIME(month(TIME) == 10));
TToct=TT(october,:);
novYr=year(TIME(month(TIME) == 11));
november=datetime(TIME(month(TIME) == 11));
TTnov=TT(november,:);
decYr=year(TIME(month(TIME) == 12));
december=datetime(TIME(month(TIME) == 12));
TTdec=TT(december,:);

ny=numel(YEAR);
nm=numel(MNTH);
nv=numel(VAR);

[ys,ms,ds]=ymd(TIME(1));
[ye,me,de]=ymd(TIME(end));
%daysMNTH=[31;28;31;30;31;30;31;31;30;31;30;31];

for k=1:ny

       TTjanA=TTjan;
       lower=datetime(YEAR(k),01,01);
       upper=datetime(YEAR(k),01,31);
       janY=isbetween(january,lower,upper);
       TTjanA(~janY,:)=[];
       DATAjan{k,1}=TTjanA;  
       isEM=isempty(DATAjan{k,1});
       if isEM==1
          DATAjan{k,1}=[];
       end
       TTfebA=TTfeb;
       lower=datetime(YEAR(k),02,01);
       upper=datetime(YEAR(k),02,28);
       febY=isbetween(febuary,lower,upper);
       TTfebA(~febY,:)=[];
       DATAfeb{k,1}=TTfebA;  
       isEM=isempty(DATAfeb{k,1});
       if isEM==1
          DATAfeb{k,1}=[];
       end
       TTmarA=TTmar;
       lower=datetime(YEAR(k),03,01);
       upper=datetime(YEAR(k),03,31);
       marY=isbetween(march,lower,upper);
       TTmarA(~marY,:)=[];
       DATAmar{k,1}=TTmarA;  
       isEM=isempty(DATAmar{k,1});
       if isEM==1
          DATAmar{k,1}=[];
       end
       TTaprA=TTapr;
       lower=datetime(YEAR(k),04,01);
       upper=datetime(YEAR(k),04,30);
       aprY=isbetween(april,lower,upper);
       TTaprA(~aprY,:)=[];
       DATAapr{k,1}=TTaprA;  
       isEM=isempty(DATAapr{k,1});
       if isEM==1
          DATAapr{k,1}=[];
       end
       TTmayA=TTmay;
       lower=datetime(YEAR(k),05,01);
       upper=datetime(YEAR(k),05,31);
       mayY=isbetween(may,lower,upper);
       TTmayA(~mayY,:)=[];
       DATAmay{k,1}=TTmayA;  
       isEM=isempty(DATAmay{k,1});
       if isEM==1
          DATAmay{k,1}=[];
       end
       TTjunA=TTjun;
       lower=datetime(YEAR(k),06,01);
       upper=datetime(YEAR(k),06,30);
       junY=isbetween(june,lower,upper);
       TTjunA(~junY,:)=[];
       DATAjun{k,1}=TTjunA;  
       isEM=isempty(DATAjun{k,1});
       if isEM==1
          DATAjun{k,1}=[];
       end
       TTjulA=TTjul;
       lower=datetime(YEAR(k),07,01);
       upper=datetime(YEAR(k),07,31);
       julY=isbetween(july,lower,upper);
       TTjulA(~julY,:)=[];
       DATAjul{k,1}=TTjulA;  
       isEM=isempty(DATAjul{k,1});
       if isEM==1
          DATAjul{k,1}=[];
       end
       TTaugA=TTaug;
       lower=datetime(YEAR(k),08,01);
       upper=datetime(YEAR(k),08,31);
       augY=isbetween(august,lower,upper);
       TTaugA(~augY,:)=[];
       DATAaug{k,1}=TTaugA;  
       isEM=isempty(DATAaug{k,1});
       if isEM==1
          DATAaug{k,1}=[];
       end
       TTsepA=TTsep;
       lower=datetime(YEAR(k),09,01);
       upper=datetime(YEAR(k),09,30);
       sepY=isbetween(september,lower,upper);
       TTsepA(~sepY,:)=[];
       DATAsep{k,1}=TTsepA;  
       isEM=isempty(DATAsep{k,1});
       if isEM==1
          DATAsep{k,1}=[];
       end
       TToctA=TToct;
       lower=datetime(YEAR(k),10,01);
       upper=datetime(YEAR(k),10,31);
       octY=isbetween(october,lower,upper);
       TToctA(~octY,:)=[];
       DATAoct{k,1}=TToctA;  
       isEM=isempty(DATAoct{k,1});
       if isEM==1
          DATAoct{k,1}=[];
       end
       TTnovA=TTnov;
       lower=datetime(YEAR(k),11,01);
       upper=datetime(YEAR(k),11,30);
       novY=isbetween(november,lower,upper);
       TTnovA(~novY,:)=[];
       DATAnov{k,1}=TTnovA;  
       isEM=isempty(DATAnov{k,1});
       if isEM==1
          DATAnov{k,1}=[];
       end
       TTdecA=TTdec;
       lower=datetime(YEAR(k),12,01);
       upper=datetime(YEAR(k),12,31);
       decY=isbetween(december,lower,upper);
       TTdecA(~decY,:)=[];
       DATAdec{k,1}=TTdecA;  
       isEM=isempty(DATAdec{k,1});
       if isEM==1
          DATAdec{k,1}=[];
       end
end    

DATAmonths{1,:}=DATAjan;
DATAmonths{2,:}=DATAfeb;
DATAmonths{3,:}=DATAmar;
DATAmonths{4,:}=DATAapr;
DATAmonths{5,:}=DATAmay;
DATAmonths{6,:}=DATAjun;
DATAmonths{7,:}=DATAjul;
DATAmonths{8,:}=DATAaug;
DATAmonths{9,:}=DATAsep;
DATAmonths{10,:}=DATAoct;
DATAmonths{11,:}=DATAnov;
DATAmonths{12,:}=DATAdec;
DATAmonth.data=DATAmonths;