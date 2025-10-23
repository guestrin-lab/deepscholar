%% This is file `iosart2x.cls'. Forked from 'iosart2c.cls' with additions from 'ios2a.cls'.
%%
%% LaTeX 2e class file for articles in the journals:
%%   Journal of Ambient Intelligence and Smart Environments (AIS)
%%   Semantic Web (SW)
%%   Web Intelligence (WEB)
%%   Journal of Clinical Speech and Language Studies (ACS)
%%   Journal of Smart Cities and Society (SCS)
%%  -
%%   Argument & Computation (AAC)
%%   Applied Ontology (AO)
%%   Asymptotic Analysis (ASY)
%%   Biorheology (BIR)
%%   Bio-Medical Materials and Engineering (BME)
%%   Biomedical Spectroscopy and Imaging (BSI)
%%   Data Science (DS)
%%   ICGA Journal (ICG)
%%   Information Services & Use (ISU)
%%   Journal of Computer Security (JCS)
%%   Journal of High Speed Networks (JHS)
%%   Journal of Integrative Neuroscience (JIN)
%%   Journal of Neutron Research (JNR)
%%   Strength, Fracture and Complexity (SFC)
%%   Technology and Health Care (THC)
%%   Journal of Future Robot Life (FRL)
%%   Fundamenta Informaticae (FI)
%%   Journal of Vestibular Research (VES)
%%  -
%%   AI Communications (AIC)
%%   Risk and Decision Analysis (RDA)
%%  -
%%   International Shipbuilding Progress (ISP)
%%  -
%%   Computability (COM)
%%  -
%%   Journal on Satisfiability, Boolean Modeling and Computation (SAT)
%%  -
%%   Clinical Hemorheology and Microcirculation (CH)
%%   Informatica (INF)
%%   NeuroRehabilitation (NRE)
%%
%% published by IOS Press, The Netherlands 
%%
%% Macros written by VTeX Ltd., Lithuania for IOS Press, The Netherlands
%% Please submit bugs or your comments to latex-support@vtex.lt
%%
%% You are free to use this class file as you see fit, provided 
%% that you do not make changes to the file. 
%% If you DO make changes, you are required to rename this file.
%%
%% It may be distributed under the terms of the LaTeX Project Public
%% License, as described in lppl.txt in the base LaTeX distribution.
%% Either version 1.0 or, at your option, any later version.
%%
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%

\def\fmt@name{iosart2x}
\def\fmt@date{2022/09/12}
\def\fmt@version{1.3.50}
\NeedsTeXFormat{LaTeX2e}[2000/01/26]
\ProvidesClass\fmt@name
              [\fmt@date\space v\fmt@version\space IOS Press]

\newif\if@restonecol \@restonecolfalse
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\newif\if@nameyear \@nameyearfalse

\RequirePackage{etoolbox}

%% class cfg load
\long\def\@if@vtex@in@house#1#2{%
  \@ifundefined{vtex@in@house}
    {#2}
    {#1}%
  }
\IfFileExists{\fmt@name-vtex.cfg}
  {%
    \xdef\vtex@in@house{1}%
    \input\fmt@name-vtex.cfg
  }{%
    \input\fmt@name.cfg
  }

\csuse{\fmt@name @load@fileinfo}

% Load flushend
\newif\if@load@flushend \@load@flushendtrue
\DeclareOption{flushend}{\@load@flushendtrue}
\DeclareOption{noflushend}{\@load@flushendfalse}

\DeclareOption{10pt}{\let\eleven@pt\relax}
\DeclareOption{11pt}{\def\eleven@pt{}}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\DeclareOption{onecolumn}{\@twocolumnfalse\@restonecoltrue}
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}%
\newif\if@rotating@loaded \@rotating@loadedfalse
\DeclareOption{rotating}{%
  \@rotating@loadedtrue
  \AtEndOfClass{%
    \RequirePackage[dvips,counterclockwise,figuresright]{rotating}%
    }%
  }
\newif\ifnumberlines \numberlinesfalse
\DeclareOption{crcready}{%
  \AtBeginDocument{%
    \gdef\rtitleheadline{}%
    \gdef\@copyright{}%
    \pagestyle{empty}%
    \numberlinesfalse
    \relax
    }%
  }%
\DeclareOption{french}{\def\article@language@opt{fr}}
\DeclareOption{german}{\def\article@language@opt{de}}
\DeclareOption{russian}{\def\article@language@opt{ru}}
\newif\if@ios@loadcjk \@ios@loadcjkfalse
\DeclareOption{loadcjk}{\@ios@loadcjktrue}
\newif\if@ios@loadcyr \@ios@loadcyrfalse
\DeclareOption{loadcyr}{\@ios@loadcyrtrue}
\newif\if@reducehelvetica \@reducehelveticatrue
\DeclareOption{normalhelvetica}{\global\@reducehelveticafalse}
\let\@copyrightowner@rights\@firstofone
\newif\if@open@access \@open@accessfalse
\DeclareOption{openaccess}{%
  \global\@open@accesstrue
  \global\let\@copyrightowner@rights\phantom
  }
\newif\if@free@to@read \@free@to@readfalse
\DeclareOption{freetoread}{\global\@free@to@readtrue}
\newif\if@autobibl \@autobibltrue
\DeclareOption{noautobibl}{\global\@autobiblfalse}

%% Numbering switches:
\newif\if@seceqn \@seceqnfalse
\DeclareOption{seceqn}{\@seceqntrue}
\newif\if@secfloat \@secfloatfalse
\DeclareOption{secfloat}{\@secfloattrue}
\newif\if@secthm
\DeclareOption{secthm}{\@secthmtrue}

%% Information about journal
\gdef\@issn{0000-0000}
\gdef\@price{00.00}
\def\copyright@price{/\$\@price}
\AtBeginDocument{%
  \if@open@access
    \def\copyright@price{}%
  \fi
  }%
\gdef\@journal{Undefined}
\def\@publisher{IOS Press}
\def\@publisheraddress{Nieuwe Hemweg 6B, 1013 BG Amsterdam, The Netherlands}
\AtBeginDocument{%
  %\ifnum\@pubyear=2020\relax
  %  \expandafter\ifstrequal\expandafter{\@JID}{SAT}{%
  %    \def\@copyrightowner{IOS Press, SAT Association and the authors.}%
  %  }{%
  %    \def\@copyrightowner{IOS Press and the authors. \@copyrightowner@rights{All rights reserved}}%
  %  }%
  %  \def\copyright@size{\footnotesize}%
  %\else
    \if@open@access
      \def\@copyrightowner{The authors. Published by IOS Press.}%
    \else
      \def\@copyrightowner{IOS Press. All rights reserved.}%
    \fi
    \def\copyright@size{\fontsize{9}{11}\selectfont}%
  %\fi
  }%

\let\do@layout\@gobble
\protected\def\dolayoutbyjid#1#2{%
  \let\do@layout\@gobble
  \@for\tmp@jid:=#1\do{%
    \ifx\tmp@jid\@jid
      \let\do@layout\@firstofone
    \fi
  }%
  \do@layout{#2}%
}
  
\gdef\ios@journal@layout{-1}%

%% >>> 0 layout >>>
\DeclareOption{ais}{%
  \gdef\@jid{ais}%
  \gdef\@JID{AIS}%
  \gdef\@issn{1876-1364}%
  \gdef\@eissn{1876-1372}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Ambient Intelligence and Smart Environments}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\ios@journal@layout{0}%
  }
  \DeclareOption{nd}{%
  \gdef\@jid{nd}%
  \gdef\@JID{ND}%
  \gdef\@issn{0}%
  \gdef\@eissn{0}%
  \gdef\@price{0}%
  \gdef\@journal{Journal Undefined}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\ios@journal@layout{0}%
  }
  
\DeclareOption{sw}{%
  \gdef\@jid{sw}%
  \gdef\@JID{SW}%
  \gdef\@issn{1570-0844}%
  \gdef\@eissn{2210-4968}%
  \gdef\@price{35.00}%
  \gdef\@journal{Semantic Web}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\ios@journal@layout{0}%
  }
\DeclareOption{web}{%
  \gdef\@jid{web}%
  \gdef\@JID{WEB}%
  \gdef\@issn{2405-6456}%
  \gdef\@eissn{2405-6464}%
  \gdef\@price{35.00}%
  \gdef\@journal{Web Intelligence}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\ios@journal@layout{0}%
  }
\DeclareOption{acs}{%
  \gdef\@jid{acs}%
  \gdef\@JID{ACS}%
  \gdef\@issn{0791-5985}%
  \gdef\@eissn{}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Clinical Speech and Language Studies}%
  \global\@twocolumntrue
  \gdef\ios@journal@layout{0}%
  }
\DeclareOption{scs}{%
  \gdef\@jid{scs}%
  \gdef\@JID{SCS}%
  \gdef\@issn{2772-3577}%
  \gdef\@eissn{2772-3585}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Smart Cities and Society}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\ios@journal@layout{0}%
  }
%% <<< 0 layout <<<

%% >>> 1 layout >>>
\DeclareOption{aac}{%
  \gdef\@jid{aac}
  \gdef\@JID{AAC}
  \gdef\@issn{1946-2166}
  \gdef\@eissn{1946-2174}%
  \gdef\@price{35.00}%
  \gdef\@journal{Argument \& Computation}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \gdef\@open@access@control{1}%
  }
\DeclareOption{ao}{%
  \gdef\@jid{ao}
  \gdef\@JID{AO}
  \gdef\@issn{1570-5838}
  \gdef\@eissn{1875-8533}%
  \gdef\@price{35.00}%
  \gdef\@journal{Applied Ontology}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \def\tagbib@style{bibstyle2}%
  \global\@nameyeartrue
  }
\DeclareOption{asy}{%
  \gdef\@jid{asy}
  \gdef\@JID{ASY}
  \gdef\@issn{0921-7134}
  \gdef\@eissn{1875-8576}%
  \gdef\@price{35.00}%
  \gdef\@journal{Asymptotic Analysis}%
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{bir}{%
  \gdef\@jid{bir}
  \gdef\@JID{BIR}
  \gdef\@issn{0006-355X}
  \gdef\@eissn{1878-5034}%
  \gdef\@price{35.00}%
  \gdef\@journal{Biorheology}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \def\tagbib@style{vancouver}%
  }
\DeclareOption{bme}{%
  \gdef\@jid{bme}
  \gdef\@JID{BME}
  \gdef\@issn{0959-2989}
  \gdef\@eissn{1878-3619}%
  \gdef\@price{35.00}%
  \gdef\@journal{Bio-Medical Materials and Engineering}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{bsi}{%
  \gdef\@jid{bsi}
  \gdef\@JID{BSI}
  \gdef\@issn{2212-8794}
  \gdef\@eissn{2212-8808}%
  \gdef\@price{35.00}%
  \gdef\@journal{Biomedical Spectroscopy and Imaging}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{ds}{%
  \gdef\@jid{ds}
  \gdef\@JID{DS}
  \gdef\@issn{2451-8484}
  \gdef\@eissn{2451-8492}%
  \gdef\@journal{Data Science}%
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \gdef\@open@access@control{1}%
  \gdef\license@type{CC@BY}%
  \gdef\copyright@type{short}%
  }
\DeclareOption{icg}{%
  \gdef\@jid{icg}
  \gdef\@JID{ICG}
  \gdef\@issn{1389-6911}
  \gdef\@eissn{2468-2438}%
  \gdef\@price{35.00}%
  \gdef\@journal{ICGA Journal}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \def\tagbib@style{bibstyle2}% IOS 2
  \global\@nameyeartrue
  \AtBeginDocument{%
    \def\copyright@size{\footnotesize}%
    \global\def\esm@skip@suppress{\dimexpr-3.5\baselineskip\relax}%
    }%
  }
\DeclareOption{isu}{%
  \gdef\@jid{isu}%
  \gdef\@JID{ISU}%
  \gdef\@issn{0167-5265}%
  \gdef\@eissn{1875-8789}%
  \gdef\@price{35.00}%
  \gdef\@journal{Information Services \& Use}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{jcs}{%
  \gdef\@jid{jcs}%
  \gdef\@JID{JCS}%
  \gdef\@issn{0926-227X}
  \gdef\@eissn{1875-8924}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Computer Security}%
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{jhs}{%
  \gdef\@jid{jhs}%
  \gdef\@JID{JHS}%
  \gdef\@issn{0926-6801}
  \gdef\@eissn{1875-8940}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of High Speed Networks}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  }
\DeclareOption{jin}{%
  \gdef\@jid{jin}%
  \gdef\@JID{JIN}%
  \gdef\@issn{0219-6352}%
  \gdef\@eissn{1757-448X}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Integrative Neuroscience}%
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{jnr}{%
  \gdef\@jid{jnr}%
  \gdef\@JID{JNR}%
  \gdef\@issn{1023-8166}%
  \gdef\@eissn{1477-2655}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Neutron Research}%
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  }
\DeclareOption{sfc}{%
  \gdef\@jid{sfc}
  \gdef\@JID{SFC}
  \gdef\@issn{1567-2069}
  \gdef\@eissn{1875-9262}%
  \gdef\@price{35.00}%
  \gdef\@journal{Strength, Fracture and Complexity}%
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{thc}{%
  \gdef\@jid{thc}%
  \gdef\@JID{THC}%
  \gdef\@issn{0928-7329}%
  \gdef\@eissn{1878-7401}%
  \gdef\@price{35.00}%
  \gdef\@journal{Technology and Health Care}%
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  }
\DeclareOption{frl}{%
  \gdef\@jid{frl}
  \gdef\@JID{FRL}
  \gdef\@issn{2589-9953}
  \gdef\@eissn{2589-9961}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Future Robot Life}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \def\tagbib@style{bibstyle2}% IOS 2
  \global\@nameyeartrue
  \AtBeginDocument{%
    \global\def\esm@skip@suppress{\dimexpr-3.5\baselineskip\relax}%
    }%
  }
\DeclareOption{fi}{%
  \gdef\@jid{fi}
  \gdef\@JID{FI}
  \gdef\@issn{0169-2968}
  \gdef\@eissn{1875-8681}%
  \gdef\@price{35.00}%
  \gdef\@journal{Fundamenta Informaticae}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \def\tagbib@style{bibstyle2}% IOS 2
  \global\@nameyeartrue
  \AtBeginDocument{%
    \global\def\esm@skip@suppress{\dimexpr-3.5\baselineskip\relax}%
    }%
  }
\DeclareOption{ves}{%
  \gdef\@jid{ves}
  \gdef\@JID{VES}
  \gdef\@issn{0957-4271}
  \gdef\@eissn{1878-6464}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal of Vestibular Research}
  \gdef\ios@journal@layout{1}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \def\tagbib@style{bibstyle2}% IOS 2
  \global\@nameyeartrue
  \AtBeginDocument{%
    \global\def\esm@skip@suppress{\dimexpr-3.5\baselineskip\relax}%
    }%
  }
%% <<< 1 layout <<<

%% >>> 2 layout >>>
\DeclareOption{aic}{%
  \gdef\@jid{aic}
  \gdef\@JID{AIC}
  \gdef\@issn{0921-7126}
  \gdef\@eissn{1875-8452}%
  \gdef\@price{35.00}%
  \gdef\@journal{AI Communications}
  \gdef\ios@journal@layout{2}%
  \global\@twocolumnfalse \global\@restonecoltrue
  }
\DeclareOption{rda}{%
  \gdef\@jid{rda}
  \gdef\@JID{RDA}
  \gdef\@issn{1569-7371}
  \gdef\@eissn{1875-9173}%
  \gdef\@price{35.00}%
  \gdef\@journal{Risk and Decision Analysis}
  \gdef\ios@journal@layout{2}%
  \global\@twocolumntrue \global\@restonecolfalse
  }
%% <<< 2 layout <<<

%% 3 layout
\DeclareOption{isp}{%
  \gdef\@jid{isp}
  \gdef\@JID{ISP}
  \gdef\@issn{0020-868X}
  \gdef\@eissn{1566-2829}%
  \gdef\@price{35.00}%
  \gdef\@journal{International Shipbuilding Progress}
  \gdef\ios@journal@layout{3}%
  \global\@twocolumnfalse \global\@restonecoltrue
  }

%% 4 layout
\DeclareOption{com}{%
  \gdef\@jid{com}
  \gdef\@JID{COM}
  \gdef\@issn{2211-3568}
  \gdef\@eissn{2211-3576}%
  \gdef\@price{35.00}%
  \gdef\@journal{Computability}
  \gdef\ios@journal@layout{4}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \AtBeginDocument{%
    \global\def\esm@skip@suppress{\dimexpr-1.5\baselineskip\relax}%
    }%
  }

%% 5 layout
\DeclareOption{sat}{%
  \gdef\@jid{sat}%
  \gdef\@JID{SAT}%
  \gdef\@issn{}%
  \gdef\@eissn{1574-0617}%
  \gdef\@price{35.00}%
  \gdef\@journal{Journal on Satisfiability, Boolean Modeling and Computation}%
  \gdef\ios@journal@layout{5}%
  \global\@twocolumnfalse \global\@restonecoltrue
  \gdef\eleven@pt{}%
  \gdef\@open@access@control{1}%
  }

%% >>> Meta layout >>>
\DeclareOption{ch}{%
  \gdef\@jid{ch}%
  \gdef\@JID{CH}%
  \gdef\@issn{1386-0291}%
  \gdef\@eissn{1875-8622}%
  \gdef\@price{35.00}%
  \gdef\@journal{Clinical Hemorheology and Microcirculation}%
  \global\@twocolumntrue
  \gdef\ios@journal@layout{0}%
  }
\DeclareOption{inf}{%
  \gdef\@jid{inf}%
  \gdef\@JID{INF}%
  \gdef\@issn{0868-4952}%
  \gdef\@eissn{1822-8844}%
  \gdef\@price{35.00}%
  \gdef\@journal{Informatica}%
  \global\@twocolumntrue
  \gdef\ios@journal@layout{0}%
  }
\DeclareOption{nre}{%
  \gdef\@jid{nre}%
  \gdef\@JID{NRE}%
  \gdef\@issn{1053-8135}%
  \gdef\@eissn{1878-6448}%
  \gdef\@price{35.00}%
  \gdef\@journal{NeuroRehabilitation}%
  \global\@twocolumntrue
  \gdef\ios@journal@layout{0}%
  }
%% <<< Meta layout <<<

%% class versions for users
\def\iosx@version{112}% initial version

\PassOptionsToPackage{fleqn}{amsmath}
\ExecuteOptions{twoside,final,openright,fleqn}

\ProcessOptions

\csuse{\fmt@name @load@dummy@layout}
\csuse{\fmt@name @load@class@version@if}
\csuse{\fmt@name @load@openaccess@check}
\csuse{\fmt@name @load@vkt@cmn@md}

\def\qq@text{TO BE UPDATED}

%% LAYOUTS
\newlength\abstract@width
\newdimen\b@author@skip
\newdimen\abstract@name@skip
\newdimen\abstract@indent@val
\newskip\b@history@skip
\def\common@layout@settings{%
  \let\ge\geqslant
  \let\geq\geqslant
  \let\le\leqslant
  \let\leq\leqslant
  \renewcommand{\Bigg}{\bBigg@{2.6}}%
  \def\copyright@sign@delim{\ }%
  \def\@copyright@delim{ -- }%
  \let\auth@style\relax
  \let\auth@fn@skip\,%
  \b@author@skip=8\p@
  \newcommand\abstractname@sep{.\ }%
  \abstract@name@skip=\z@
  \abstract@width=\textwidth
  \def\abstract@indent{%
    \noindent
    \parindent=1em%
    }%
  \def\keywords@indent{\parindent=\z@}%
  \let\keywordsname@fmt\@firstofone
  \newcommand\keywordsname@sep{:\ }%
  \def\keywords@size{\abstract@size}%
  \def\history@size{\footnotesize\baselineskip=12pt}%
  \b@history@skip=12pt%
  %
  \def\accepted@postfix{\par}%
  \def\acceptedby@postfix{\par}%
  \def\communicated@postfix{\par}%
  \def\published@postfix{\par}%
  \def\received@postfix{\par}%
  \def\revised@postfix{\par}%
  \def\submitted@postfix{\par}%
  %
  \def\dedicated@postfix{\par\egroup}%
  %
  \def\history@postfix{}%
  \let\upd@hist@prefices\relax
  \def\tablefont{\footnotesize}%
  }
\def\common@settings@from@ios@two@a{%
  \def\tfrac##1##2{{\textstyle\frac{##1}{##2}}}
  \def\dfrac##1##2{{\displaystyle\frac{##1}{##2}}}
  \def\dvt{\colon\ }
  \def\dvtx{\nobreak\,\colon\relax}
  \AtEndOfClass{\mathindent=2em}
  \let\rtitleheadline@size\footnotesize
  \b@pretitle@skip=6pt
  \b@author@skip=12\p@
  \abstract@skip=18\p@
  \keywords@skip=6\p@
  \keywords@noabs@skip=18\p@
  \b@section@skip=24pt plus6pt minus6pt%
  \tolerance=1600
  \hbadness=1600
  \interdisplaylinepenalty=100
  \setlength{\skip\footins}{12\p@ \@plus 6\p@}
  \skip\@mpfootins=\skip\footins
  \def\biblio@clubpenalty{8000}
  \def\biblio@widowpenalty{7000}
  \def\NAT@penalty{\penalty=50}
  \setlength{\bibhang}{12pt}
  \bibsep=\z@ \@plus.5\p@
  \appto\bibpreamble{%
    \leftmargin=12\p@
    \labelsep=5\p@
    }
  \hfuzz=0.1\p@
  }
\@namedef{init@ios@journal@0@layout}{%
  \AtEndOfClass{%
    \mathindent=\z@
    }%
  \ifnum\jobstart@date@info>20211020\relax
    \interdisplaylinepenalty=100
  \fi
  }%
\@namedef{init@ios@journal@1@layout}{% B type from ios2a
  \common@settings@from@ios@two@a
  \brokenpenalty=100
  \clubpenalty=4000
  \widowpenalty=4000
  \adjdemerits=10000
  \doublehyphendemerits=10000
  \predisplaypenalty=10000 % Breaking before a math display.
  \def\abstract@size{\footnotesize}
  \def\proof@topsep{11pt plus 3pt}
  \def\@listI{%
    \leftmargin\leftmargini
    \parsep=0\p@ \@plus2\p@ \@minus\p@
    \topsep=9\p@ \@plus2\p@ \@minus2\p@
    \partopsep=\p@
    \itemsep=1\p@ \@plus.5\p@ \@minus1\p@
    }
  \let\@listi\@listI
  \@listi
  }
\@namedef{init@ios@journal@2@layout}{% A type from ios2a
  \common@settings@from@ios@two@a
  \AtEndOfClass{%
    \mathindent=\z@
    }
  \if@twocolumn
    \setlength\tablewidth{217.66pt}
  \fi
  \leftmargini=25pt
  \leftmargin=\leftmargini
  \leftmarginv=10pt
  \leftmarginvi=10pt
  \setlength\labelwidth{\leftmargini}
  \addtolength\labelwidth{-\labelsep}
  \b@pretitle@skip=16.5pt
  }
\@namedef{init@ios@journal@3@layout}{% C type from ios2a
  \common@settings@from@ios@two@a
  \@nameuse{init@ios@journal@1@layout}% B type from ios2a
  \@@topskip=12pt
  \b@section@skip=36pt plus2pt minus6pt
  \def\pretitle@size{\normalsize}
  \def\title@size{\Large\bfseries\mathversion{bold}}
  \def\subtitle@size{\normalsize\itshape}
  \def\address@size{\footnotesize\itshape}
  \let\rtitleheadline@size\scriptsize
  \let\rtitleheadline@pagesize\scriptsize
  \def\abstract@indent{\parindent=1em}
  \def\abstractname{}
  \def\abstractname@sep{}
  \def\proof@topsep{11pt plus 3pt}
  \AtBeginDocument{%
    %\ifnum\@pubyear=2020\relax
    %  \def\copyright@size{\footnotesize}%
    %\else
      \def\copyright@size{\fontsize{8}{10}\selectfont}%
    %\fi
    }%
  }
\@namedef{init@ios@journal@4@layout}{% D type from ios2a
  \common@settings@from@ios@two@a
  \setlength\footnotesep{9.4\p@}%
  \setlength{\skip\footins}{10\p@ \@plus 2\p@ \@minus 2\p@}%
  \skip\@mpfootins=\skip\footins
  \def\initsline@vskip{10ex}
  \def\footnoterule{%
    \kern-3\p@
    \hrule width\textwidth
    \kern 2.6\p@
    }
  \setlength\floatsep       {20\p@ \@plus 2\p@ \@minus 0\p@}
  \setlength\textfloatsep   {17\p@ \@plus 4\p@ \@minus 0\p@}
  \setlength\intextsep      {12\p@ \@plus 2\p@ \@minus 0\p@}
  \setlength\dblfloatsep    {12\p@ \@plus 2\p@ \@minus 0\p@}
  \setlength\dbltextfloatsep{15\p@ \@plus 4\p@ \@minus 0\p@}
  \@fpsep=8pt plus 2fil
  \@dblfpsep=8pt plus 2fil
  \renewcommand\figurename{Figure}
  \def\fnum@figure{\textbf{\figurename~\thefigure.}}
  \ifnum\jobstart@date@info>20211020\relax\else
    \brokenpenalty=10000
  \fi
  \clubpenalty=4000
  \widowpenalty=4000
  \adjdemerits=10000
  \doublehyphendemerits=10000
  \b@pretitle@skip=-10\p@
  \def\title@size{%
    \fontsize{14.4}{18}\selectfont
    \bfseries
    \mathversion{bold}%
    }%
  \advance\@@topskip by -3.5\p@
  \a@title@skip=\z@
  \b@author@skip=9\p@
  \def\authors@size{\normalsize}
  \def\address@size{\footnotesize}
  \let\ioscom@author@optarg\author@optarg
  \def\author@optarg[##1]##2{%
    \vskip3pt plus .1\p@ minus .1\p@
    \ioscom@author@optarg[##1]{##2}%
    \ifx.##1.%
      \ioscom@printaddress{}%
    \else
      \@for\@tempa:=##1\do{%
        \ioscom@printaddress\@tempa
        }%
    \fi
    \endgraf
    }%
  \csuse{\fmt@name @load@ioscom@printaddress}
  \let\add@addressref\@gobble
  \def\address{%
    \@ifnextchar[%
      {\ioscom@address}
      {\ioscom@address[]}%
    }%
  \def\ioscom@address[##1]##2{%
    \def\ioscom@tmp{##2}%
    \protected@write\@auxout{}{%
      \string\global\string\@namedef{ioscom@##1@address}{\expandafter\strip@prefix\meaning\ioscom@tmp}%
      }%
    }%
  \def\abstract@size{\small}
  \abstract@skip=16pt plus .5pt minus .5pt
  \keywords@skip=13pt plus .5pt minus .5pt
  \def\keywordsname@fmt##1{\textbf{##1}\textbf}%
  \b@section@skip=15pt plus1pt minus1pt
  \g@addto@macro\close@fm{%
    \global\@afterindentfalse
    \aftergroup\@afterheading
    }
  \let\bibfont\normalsize
  \renewcommand\NAT@sep{,\NAT@penalty}
  \def\section@style{\fontsize{13}{15}\bfseries\selectfont\mathversion{bold}\nohyphen\raggedright}
  \def\sectionpart@style{\fontsize{13}{15}\bfseries\selectfont\mathversion{bold}\nohyphen\raggedright}
  \def\subsection@style{\fontsize{11}{13}\bfseries\selectfont\nohyphen\raggedright}
  \def\subsubsection@style{\normalsize\nohyphen\raggedright}
  \def\paragraph@style{\normalsize\itshape\raggedright}
  \def\subparagraph@style{\paragraph@style}
  \def\section@beforeskip{15\p@ \@plus -4\p@}
  \def\sectionpart@beforeskip{-10\p@ \@plus -3\p@ \@minus -1\p@}
  \def\subsection@beforeskip{-15\p@ \@plus -3\p@}
  \def\subsubsection@beforeskip{-15\p@ \@plus -3\p@}
  \def\paragraph@beforeskip{-6\p@ \@plus -3\p@}
  \def\subparagraph@beforeskip{-0.1\p@}
  \def\section@afterskip{5\p@ \@plus 0.5\p@}
  \def\sectionpart@afterskip{4\p@ \@plus 2\p@ \@minus 0.5\p@}
  \def\subsection@afterskip{3\p@ \@plus 0.3\p@}
  \def\subsubsection@afterskip{3\p@ \@plus 0.3\p@}
  \def\paragraph@afterskip{-1em}
  \def\subparagraph@afterskip{-1em}
  \def\subparagraph@indent{1em}
  \let\rtitleheadline@size\scriptsize
  \AtBeginDocument{%
    %\ifnum\@pubyear=2020\relax
    %  \def\copyright@size{\scriptsize}%
    %\else
      \def\copyright@size{\fontsize{9}{11}\selectfont}%
    %\fi
    }%
  \def\copyright@vskip{\vskip36.2pt}
  \def\@listI{%
    \leftmargin=\leftmargini
    \parsep=0\p@ \@plus2\p@ \@minus\p@
    \topsep=9\p@ \@plus2\p@ \@minus2\p@
    \partopsep=\p@
    \itemsep=1\p@ \@plus.5\p@ \@minus1\p@
    }
  \let\@listi\@listI
  \@listi
  \def\mathend@qed{\hbox to\hsize{\hfill$\Box$}}
  \def\qed{%
    \ifmmode
      \hfill\Box
    \else
      \leavevmode
      \unskip\penalty9999
      \hbox{}\nobreak\hfill\nobreak$\Box$%
    \fi
    }
  \def\proof@topsep{11pt plus 3pt}
  \AtEndOfClass{%
    \urlstyle{tt}%
    \let\ioscom@href\href
    \def\href##1##2{\ioscom@href{##1}{\texttt{##2}}}
    }
  }

\newif\if@make@lua@auth@ \@make@lua@auth@false
\ifdefined\HCode\else
  \ifnum\ios@journal@layout=5\relax
    \@make@lua@auth@true
  \fi
\fi

\let\vtx@directlua\@gobble
\ifdefined\directlua
  \let\vtx@directlua\directlua
\fi

\vtx@directlua{%
  tab_chr = string.char(9)            % "\t"
  eol_chr = string.char(10)           % "\n"
  cr_chr = string.char(10)            % "\r"
  hash_chr = string.char(35)          % "#"
  perc_chr = string.char(37)          % "%"
  bslash_chr = string.char(92)        % "\\"
  underscore_chr = string.char(95)    % "_"
  ampersand_chr = string.char(38)     % "&"
  }%

\@namedef{init@ios@journal@5@layout}{% SAT
  \hoffset=-0.88in
  \voffset=-1.8mm
  %
  \newcount\pdfshellescape
  \pdfshellescape=\@ne
  \if@make@lua@auth@
    \vtx@directlua{%
      \short@journal_author = {}          % array of authors
                                          % index -- author order number
                                          % value -- record with following fields:
                                          %     .fnms -- first name(s), not a value, just a marker of the presence: 'x' or nil
                                          %     .thanks -- corresponding author note, not a value, just a marker of the presence: 'x' or nil
                                          %     .eads -- csv list of email labels \ead@labels, collected from optional parameters of \ead[] -- 'e1', 'e1,e2', etc.
                                          %     .addrefs -- an array of references to addresses: {'A'}, {'B', 'C'}, etc. -- values from optional parameter of \author[]
                                          %     .done -- author already printed: true or nil
                                          %   unused:
                                          %     .locnotes -- value of \t@loc@notes after processing with \proc@elem
      %
      \short@journal_address = {}         % array of addresses
                                          % index -- address order number
                                          % value -- record with following fields:
                                          %     .addref -- reference to the addresses: 'A', 'B', 'C', etc. -- values from optional parameter of \address[]
                                          %     .done -- address already printed: true or nil
                                          %   unused:
                                          %     .locnotes -- value of \t@loc@notes after processing with \proc@elem
      %
      \short@journal_addr_map = {}        % back mapping between addrefs and addres indices
                                          % index -- addref -- reference to the addresses: 'A', 'B', 'C', etc. -- values from optional parameter of \address[]
                                          % value -- address index (order number)
      }%
  \fi
  \def\bfseries{\fontseries{bx}\selectfont}%
  \csname init@ios@journal@1@layout\endcsname
  \def\rtitleheadline@size{\scriptsize}%
  \def\@runsep{}%
  \def\@evenhead{\small\scshape\hfill\@runauthor\hfill}%
  \def\@oddhead{\small\scshape\hfill\@runtitle\hfill}%
  \def\@evenfoot{\small\rlap{\thepage}\hfill\includegraphics[scale=.3]{sat1cim}\hfill}%
  \def\@oddfoot{\small\hfill\includegraphics[scale=.3]{sat1cim}\hfill\llap{\thepage}}%
  \AtBeginDocument{%
    %\ifnum\@pubyear=2020\relax
    %  \def\copyright@size{\scriptsize}%
    %\else
      \def\copyright@size{\fontsize{8}{10}\selectfont}%
    %\fi
    \expandafter\ifstrequal\expandafter{\@JID}{SAT}{%
      %\ifnum\@pubyear=2020\relax
      %  \def\copyright@sign@delim{}%
      %  \def\@copyright@delim{\ }%
      %  \def\standard@copyright@line{\copyright@statement}%
      %\else
        \def\copyright@sign@delim{\ }%
        \def\@copyright@delim{\ --\ }%
        \def\standard@copyright@line{\@eissn\copyright@price\ \copyright@statement}%
      %\fi
      }{}%
    }%
  \@@topskip=-7\p@
  \def\title@size{\center\Large\bfseries\mathversion{bold}}%
  \b@author@skip=7\p@
  \def\authors@size{\normalsize}%
  \def\auth@style{\bfseries}%
  \let\auth@fn@skip\relax
  \def\address@size{\small\itshape}%
  \renewcommand\abstractname@sep{}%
  \abstract@name@skip=2\p@
  \abstract@width=413\p@
  \abstract@indent@val=20\p@%\dimexpr(\textwidth-\abstract@width)/2\relax
  \def\abstract@indent{%
    \hangindent=\abstract@indent@val%\leftmargin=\abstract@indent@val
    \parindent=\dimexpr\parindent@value+\hangindent\relax
    }%
  \def\abstract@size{\small}%
  \keywords@skip=12\p@
  \def\keywords@indent{%
    \abstract@indent
    \parindent=\hangindent
    }%
  \def\keywordsname@fmt##1{{\scshape ##1}}%
  \renewcommand\keywordsname@sep{{\normalfont :}\hskip 10\p@}%
  \def\keywords@size{\abstract@size\itshape}%
  \def\history@size{\footnotesize\itshape}%
  %
  \def\accepted@postfix{}%
  \def\acceptedby@postfix{}%
  \def\communicated@postfix{}%
  \def\dedicated@postfix{}%
  \def\published@postfix{}%
  \def\received@postfix{}%
  \def\revised@postfix{}%
  \def\submitted@postfix{}%
  %
  \ifdefined\HCode\else
    \def\upd@hist@prefices{%
      \def\accepted@prefix{; accepted~}%
      \def\acceptedby@prefix{; accepted~by:~}%
      \def\communicated@prefix{; communicated~}%
      \def\dedicated@prefix{; dedicated~}%
      \def\published@prefix{; published~}%
      \def\received@prefix{; received~}%
      \def\revised@prefix{; revised~}%
      \def\submitted@prefix{; submitted~}%
      }%
  \fi
  \b@section@skip=4\p@
  \def\section@beforeskip{18\p@}%
  \def\section@afterskip{10\p@}%
  \def\subsection@beforeskip{16\p@}%
  \def\subsection@afterskip{6\p@}%
  \def\subsubsection@beforeskip{6\p@}%
  \def\subsubsection@afterskip{6\p@}%
  \def\section@style{\fontsize{12}{14}\selectfont\bfseries\mathversion{bold}\nohyphen\raggedright}%
  \def\subsection@style{\normalsize\bfseries\mathversion{bold}\nohyphen\raggedright}%
  \def\subsubsection@style{\normalsize\selectfont\scshape\mathversion{bold}\nohyphen\raggedright}%
  \renewcommand\section{%
    \@startsection{section}{1}{\z@}%
      {-\section@beforeskip}%
      {\section@afterskip}%
      {\section@style}%
    }
  \renewcommand\subsection{%
    \@startsection{subsection}{2}{\subsection@indent}%
      {-\subsection@beforeskip}%
      {\subsection@afterskip}%
      {\subsection@style}%
    }
  \renewcommand\subsubsection{%
    \@startsection{subsubsection}{3}{\z@}%
      {-\subsubsection@beforeskip}%
      {\subsubsection@afterskip}%
      {\subsubsection@style}%
    }
  %
  \renewcommand\figurename{Figure}%
  \global\def\fnum@figure{\normalsize\textbf{\figurename~\thefigure.}\small}
  \global\def\fnum@table{\normalsize\textbf{\tablename~\thetable.}\small}
  \global\long\def\@maketablecaption##1##2{%
    \setbox\@tempboxa\hbox{##1 ##2}%
    \ifdim\wd\@tempboxa>\hsize
      \unhbox\@tempboxa
      \par
    \else
      \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
    \fi
    \vskip\belowcaptionskip
    }%
  \long\def\@makefigurecaption##1##2{%
    \vskip\abovecaptionskip
    \setbox\@tempboxa\hbox{##1 ##2}%
    \ifdim\wd\@tempboxa>\hsize
      \unhbox\@tempboxa
      \par
    \else
      \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
    \fi
    \vskip\belowcaptionskip
    }%
  \setlength\belowcaptionskip{10\p@}%
  \def\tablefont{\normalsize}%
  %
  \AtBeginDocument{%
    \def\proof@topsep{\z@}
    }%
  \def\proofname{Proof}%
  \def\@proof@label@sep{:}%
  %
  \let\bibfont\normalsize
  \bibsep=9\p@\@plus.5\p@\@minus.5\p@
  %
  \definecolor{NavyBlue}{rgb}{0.1,0.45,0.82}%
  \definecolor{WildStrawberry}{rgb}{0.93,0.23,0.43}%
  \definecolor{Maroon}{rgb}{0.5,0,0}%
  \AtBeginDocument{%
    \hypersetup{%
      linkcolor=Maroon,
      citecolor=NavyBlue,
      urlcolor=WildStrawberry
      }%
    }%
  \AtEndOfClass{%
    \urlstyle{tt}%
    \let\ioscom@href\href
    \def\href##1##2{\ioscom@href{##1}{\texttt{##2}}}
    }
  }
\AtBeginDocument{%
  \ifnum\ios@journal@layout=5\relax
    \renewenvironment{proof}[1][\proofname]%
      {%
  %      \par
        \pushQED{\qed}%
        \normalfont
        \topsep=\proof@topsep
        \relax
        \trivlist
        \item[%
          \hskip\labelsep
          \bfseries
          #1\@proof@label@sep]%
        \ignorespaces
      }{%
        \hfill\null
        \popQED
        \endtrivlist
        \@endpetrue
        \vskip6\p@
      }%
  \fi
  }

\ifdefined\@open@access@control
  \AtBeginDocument{\mandatory@open@access@option@error}%
\fi

%% Journal-specific settings
\def\init@ios@journal@ASY@settings{%
  \gappto\@articletype@prehook{%
    \global\let\ios@pages@trimbox\ios@pages@trimbox@papersize
    }%
  }
\def\init@ios@journal@COM@settings{%
  \appto\before@printead@hook{%
    \let\ead@prefixsep\relax
    \let\ead@prefix@fax\relax
    \let\ead@prefix@url\relax
    \let\ead@prefix@email\relax
    \let\ead@prefix@phone\relax
    \let\ead@prefix@orcid\relax
    \let\ead@prefix@m@fax\relax
    \let\ead@prefix@m@url\relax
    \let\ead@prefix@m@email\relax
    \let\ead@prefix@m@phone\relax
    \let\ead@prefix@m@orcid\relax
    }%
  }
\def\init@ios@journal@DS@settings{%
  \def\bisbn##1{\href{http://www.isbnsearch.org/isbn/##1}{##1}}%
  \let\review\dsreview
  \let\endreview\enddsreview
  }
\def\init@ios@journal@ICG@settings{%
  \def\prepress@id@line@text{%
    \hfill p.~\number\@@@sheetcount/\totalpages\fp@id@line@addon
    }
  \def\title@size{\LARGE}%
  \csuse{\fmt@name @ICG@section@style}%
  \def\subsection@style{\fontsize{10}{12}\selectfont\nohyphen\raggedright}%
  \long\def\bibfont{%
    \normalsize
    \baselineskip=\dimexpr1\baselineskip\@plus.5\p@\relax
    }%
  \bibsep=5\p@\@plus.5\p@\@minus.5\p@
  \parskip=5\p@\@plus.5\p@\@minus.5\p@
  \parindent=\z@
  \setlength{\bibhang}{\z@}%
  %% always centered float captions
  \long\def\@makefigurecaption##1##2{%
    \footnotesize
    \vskip\abovecaptionskip
    \bgroup
      \centering
      ##1 ##2%
      \par
    \egroup
    }%
  \long\def\@maketablecaption##1##2{%
    \begingroup
      \footnotesize
      \global\setbox\@tempboxa\hbox{##2}%
    \endgroup
    \centering
    \hbox to \hsize{\footnotesize\hss##1\hss}%
    \par
    \ifdim\wd\@tempboxa>\tablewidth
      \parbox[t]{\tablewidth}{%
        \footnotesize
        \centering
        \mbox{}%
        ##2%
        \vphantom{Ay}%
        \par
        }%
    \else
      \hbox to\hsize{\hfill\box\@tempboxa\vphantom{Ay}\hfill}%
    \fi
    \vskip\belowcaptionskip
    }%
    \setlength\belowcaptionskip{10\p@}
    \AtEndOfClass{
      \let\sidewaystable\undefined
      \cslet{sidewaystable*}\undefined
      }
  }
\let\init@ios@journal@FRL@settings\init@ios@journal@ICG@settings
\def\init@ios@journal@INF@settings{
  \def\theaddress{\arabic{address}}
  \def\@copyrightowner{Vilnius University}
  \def\@publisher{Vilnius University Institute of Mathematics and Informatics}
  \def\@publisheraddress{Akademijos 4, LT-08663 Vilnius, Lithuania}
  }
\def\init@ios@journal@SW@settings{%
  \b@author@skip=-2\p@
  \dsreview@postskip=6\p@
  \AtEndOfClass{\mathindent=2em}%
  }
\def\init@ios@journal@WEB@settings{%
  \AtEndOfClass{\mathindent=2em}%
  }
\def\init@ios@journal@SAT@settings{%
  \gappto\@articletype@prehook{%
    \global\let\ios@pages@trimbox\ios@pages@trimbox@papersize
    }%
  \appto\before@printead@hook{%
    \let\ead@prefixsep\relax
    \let\ead@prefix@fax\relax
    \let\ead@prefix@url\relax
    \let\ead@prefix@email\relax
    \let\ead@prefix@phone\relax
    \let\ead@prefix@m@fax\relax
    \let\ead@prefix@m@url\relax
    \let\ead@prefix@m@email\relax
    \let\ead@prefix@m@phone\relax
    }%
  }
\def\init@ios@journal@AIC@settings{%
    \AtEndOfClass{\mathindent=2em}%
  }
\def\init@ios@journal@AIS@settings{%
    \AtEndOfClass{\mathindent=2em}%
  }
\def\init@ios@journal@SCS@settings{%
  \AtEndOfClass{\mathindent=2em}%
  }
  
%% Fonts
\ifnum\ios@journal@layout>\z@
  \def\@xipt{11}
  \def\@xivpt{14}
  \def\@xviipt{17}
  \def\@xviiipt{18}
  \def\@xxpt{20}
  \def\@xxivpt{24}
  \def\@xxvpt{25}
\fi

\@ifundefined{eleven@pt}
  {%
    \typeout{Ten point}
    \ifnum\ios@journal@layout>\z@
      \renewcommand\normalsize{%
        \@setfontsize\normalsize\@xpt{12pt plus .5\p@ minus .5\p@}%
        \abovedisplayskip=12\p@\@plus2\p@\@minus2\p@
        \belowdisplayskip=\abovedisplayskip
        \let\@listi\@listI
        }%
      \newcommand\small{%
        \@setfontsize\small\@ixpt{11}%
        \abovedisplayskip=11\p@\@plus2\p@
        \belowdisplayskip=\abovedisplayskip
        \def\@listi{%
          \leftmargin=\leftmargini
          \topsep=5\p@\@plus2\p@\@minus2\p@
          \parsep=\p@\@plus\p@\@minus\p@
          \itemsep=\parsep
          }%
        }%
      \newcommand\footnotesize{%
        \@setfontsize\footnotesize\@viiipt\@xpt
        \abovedisplayskip=10\p@\@plus2\p@\@minus2\p@
        \def\@listi{%
          \leftmargin=\leftmargini
          \topsep=5\p@\@plus\p@\@minus\p@
          \parsep=\p@\@plus\p@\@minus\p@
          \itemsep=\parsep
          }%
        \belowdisplayskip=\abovedisplayskip
        }%
    \else
      \renewcommand\normalsize{%
        \@setfontsize\normalsize\@xpt{12pt plus .5\p@ minus .1\p@}%
        \abovedisplayskip=12\p@\@plus2\p@\@minus2\p@
        \abovedisplayshortskip=\abovedisplayskip
        \belowdisplayshortskip=\abovedisplayskip
        \belowdisplayskip=\abovedisplayskip%
        \let\@listi\@listI
        }%
      \newcommand\small{%
        \@setfontsize\small\@ixpt\@xipt%
        \abovedisplayskip=11\p@\@plus3\p@
        \abovedisplayshortskip=11\p@\@plus1\p@\@minus1\p@
        \belowdisplayshortskip=11\p@\@plus1\p@\@minus1\p@
        \def\@listi{%
          \leftmargin=\leftmargini
          \topsep=5\p@\@plus2\p@\@minus2\p@
          \parsep=\z@
          \itemsep=\parsep
          }%
        \belowdisplayskip=\abovedisplayskip
        }%
      \newcommand\footnotesize{%
        \@setfontsize\footnotesize\@viiipt\@xpt%
        \abovedisplayskip=10\p@\@plus3\p@
        \abovedisplayshortskip=10\p@\@plus1\p@\@minus1\p@
        \belowdisplayshortskip=10\p@\@plus1\p@\@minus1\p@
        \def\@listi{%
          \leftmargin=\leftmargini
          \topsep=4\p@\@plus2\p@\@minus2\p@
          \parsep=\z@
          \itemsep=\parsep
          }%
        \belowdisplayskip=\abovedisplayskip
        }%
    \fi
    \newcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
    \newcommand\tiny{\@setfontsize\tiny\@vipt\@viipt}
    \newcommand\large{\@setfontsize\large\@xipt{13}}
    \newcommand\Large{\@setfontsize\Large\@xivpt{18}}
    \newcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
    \newcommand\huge{\@setfontsize\huge\@xxpt{25}}
    \newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
  }{%
    \typeout{Eleven point}
    \renewcommand\normalsize{%
      \@setfontsize\normalsize\@xipt{13pt plus .5\p@ minus .1\p@}%
      \abovedisplayskip=13\p@\@plus3\p@\@minus3\p@
      \abovedisplayshortskip=\abovedisplayskip
      \belowdisplayshortskip=\abovedisplayskip
      \belowdisplayskip=\abovedisplayskip
      \let\@listi\@listI
      }%
    \newcommand\small{%
      \@setfontsize\small\@xpt\@xiipt
      \abovedisplayskip=6.5\p@\@plus3\p@
      \abovedisplayshortskip=6.5\p@\@plus1\p@\@minus1\p@
      \belowdisplayshortskip=6.5\p@\@plus1\p@\@minus1\p@
      \def\@listi{%
        \leftmargin=\leftmargini
        \topsep=6\p@\@plus2\p@\@minus2\p@
        \parsep=\z@
        \itemsep=\parsep
        }%
      \belowdisplayskip=\abovedisplayskip
      }%
    \newcommand\footnotesize{%
      \@setfontsize\footnotesize\@ixpt{10}%
      \abovedisplayskip=6.5\p@\@plus3\p@
      \abovedisplayshortskip=6.5\p@\@plus1\p@\@minus1\p@
      \belowdisplayshortskip=6.5\p@\@plus1\p@\@minus1\p@
      \def\@listi{%
        \leftmargin=\leftmargini
        \topsep=4\p@\@plus2\p@\@minus2\p@
        \parsep=\z@
        \itemsep=\parsep
        }%
      \belowdisplayskip=\abovedisplayskip
      }%
    \newcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
    \newcommand\tiny{\@setfontsize\tiny\@vipt\@viipt}
    \newcommand\large{\@setfontsize\large\@xiipt{14}}
    \newcommand\Large{\@setfontsize\Large\@xivpt{18}}
    \newcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
    \newcommand\huge{\@setfontsize\huge\@xxpt{25}}
    \newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
  }
\normalsize

\ifdefined\HCode
  \vtx@directlua{%
    utf = require 'utf8ex'
    }%
\fi

\usepackage{ifpdf}

\ifdefined\HCode
  \def\textzh#1{%
    \vtx@directlua{%
      local par1 = "#1"
      local wcharr = utf.decode_u8(par1)
      local defcmd = bslash_chr .. 'global' .. bslash_chr .. 'def' .. bslash_chr .. 'txtzh{'
      for _, wch in pairs(wcharr) do
        defcmd = defcmd .. string.format(ampersand_chr .. bslash_chr .. hash_chr .. 'x' .. perc_chr .. '04x;', wch)
      end
      defcmd = defcmd .. '}'
      tex.print(defcmd)
      }%
    \HCode{\txtzh}%
    }%
  \let\textjapanese\textzh
  \let\textrussian\textzh
  \AtBeginDocument{%
    \def\AA{\HCode{<mi mathvariant="normal">&\#x00C5;</mi>}}%
    }%
\else
  \if@ios@loadcjk
    \RequirePackage[no-math]{fontspec}
    \ifpdf\else
      \usepackage{luafonts}
    \fi
    \newfontfamily\textzhfamily{SimSun.ttf}
    \def\textzh#1{{\textzhfamily #1}}
    \newfontfamily\textjapanesefamily{MS-PMincho.ttf}
    \def\textjapanese#1{{\textjapanesefamily #1}}
  \fi
  \if@ios@loadcyr
    \RequirePackage[no-math]{fontspec}
    \ifpdf\else
      \usepackage{luafonts}
    \fi
    \input hyph-ru.tex
    \newfontfamily\textrussianfamily[
      Extension=.otf,
      UprightFont=*-regular,
      ]{old-texgyretermes}
    \def\textrussian#1{{\textrussianfamily #1}}
  \fi
  \AtBeginDocument{%
    \def\AA{%
      \ifmmode
        \mathring{\mathrm{A}}%
      \else
        \r A%
      \fi
      }%
    }%
\fi

%% Customization of fonts
\renewcommand\sldefault{it}
\renewcommand\bfdefault{b}
\let\slshape\itshape

%% Text dimensions
\newlength\parindent@value
\setlength\parindent{10\p@}
\ifcase\ios@journal@layout
  \setlength\textwidth{460\p@}% 162mm
\or
  \setlength\textwidth{160mm}%
\or
  \setlength\textwidth{162mm}%
\or
  \setlength\textwidth{118mm}%
\or
  \setlength\textwidth{162mm}%
  \setlength\parindent{5.5mm}%
\or
  \setlength\textwidth{6.0 true in}% 152.4mm
  \setlength\parindent@value{18\p@}%
  \setlength\parindent{\parindent@value}%
\fi

\setlength\textheight{50\baselineskip}
\advance\textheight by \topskip
\ifnum\ios@journal@layout=\@ne
  \setlength\textheight{206mm}% 42*13 lines
\fi
\ifnum\ios@journal@layout=\thr@@
  \setlength\textheight{180mm}%
\fi
\ifnum\ios@journal@layout=4
  \addtolength\textheight{-15.1\p@}%
\fi
\ifnum\ios@journal@layout=5
  \setlength\textheight{8.5 true in}% 215.9 mm
\fi

\def\init@ios@journal@ICG@textframe{%
  \setlength\textwidth{156mm}%
  %% 47*baselineskip = 47*13 = 611pt
  \setlength\textheight{\the\dimexpr 611pt + \topskip\relax}%
  }
\let\init@ios@journal@FRL@textframe\init@ios@journal@ICG@textframe
\@nameuse{init@ios@journal@\@JID @textframe}

\setlength\columnsep{26\p@}  %9mm 
\ifnum\ios@journal@layout>\@ne
  \setlength\columnsep{9mm}%
\fi
\setlength\columnwidth{217\p@}
\def\footnoterule@width{\the\columnwidth}
\setlength\columnseprule{0\p@}

\csuse{\fmt@name @load@vtexoutputpage}

\gappto\PROOF{%
  \AtBeginDocument{%
    \ifnum\@volume=\m@ne
      \ifnum\@issue=\m@ne
        \xdef\prepress@stage{1}%
        \protected@write\@auxout{}{\string\xdef\string\prepress@stage{1}}%
      \fi
    \fi
    }%
  }
\AtBeginDocument{%
  \csnumdef{totalpages}{\@lastpage -\@firstpage + 1}%
  }
\let\@id@line@text\@id@line
\def\fp@id@line@addon{}

\def\prepress@id@line@text{%
  [\@articletype]%
  \hfill p.~\number\@@@sheetcount/\totalpages\fp@id@line@addon
  }
\def\vtex@id@line@text{%
  \@JID\ (\fmt@name\ \fmt@date\ v\fmt@version)%
  \hfill\jobname.tex\ \TODAY\ \timenow\hfill 
  [\@articletype]\hfill p.~\number\@@@sheetcount/\totalpages
  }
\appto\HPROOF{%
  \let\@id@line@text\vtex@id@line@text
  \def\fp@id@line@addon{%
    \\PRETITLE:\space
    {\usefont{T1}{lmtt}{m}{n}\fontsize{8}{8}\selectfont\@pretitle}%
    }%
  }
\gappto\PROOF{%
  \AtBeginDocument{%
    \let\@id@line@text\vtex@id@line@text
    \def\fp@id@line@addon{%
      \\PRETITLE:\space
      {\usefont{T1}{lmtt}{m}{n}\fontsize{8}{8}\selectfont\@pretitle}%
      }%
    \@ifundefined{prepress@stage}
      {}
      {%
        \let\fp@id@line@addon\relax
        \let\@id@line@text\prepress@id@line@text
      }%
    }%
  }

\@ifundefined{idline@raise}{\def\idline@raise{}}{}

\def\journal@idline{%
  \idline@raise
  \hbox{%
    % \special{vtex:bmc.artifact.idline}% for dvi differ
    \parbox[t]{\textwidth}{%
      \@id@line@text
      \fp@id@line@addon
      \global\let\fp@id@line@addon\relax
      }%
    % \special{vtex:emc.artifact.idline}%
    }%
  }
\def\@id@line{\journal@idline}%
\AtBeginDocument{%
  \ifdefined\suppress@info
    \let\journal@idline\relax
  \fi
  }%
\def\vtexed#1{}
\g@addto@macro\PROOF\SETGRID

%% Centering on A4 page
\hoffset=-1in
\voffset=-1.15mm
\def\paper@width{210mm}\paperwidth=210mm
\def\paper@height{297mm}\paperheight=297mm
\@tempdima=\paper@width
\advance\@tempdima by-\textwidth
\divide\@tempdima  by2

% HEADS:
\setlength\headheight{12\p@}
\setlength\headsep   {15\p@}
\setlength\topskip   {10\p@}
\setlength\footskip  {25\p@}
\setlength\maxdepth  {.5\topskip}

\ifnum\ios@journal@layout=4
  \setlength\headheight{5.6\p@}
  \setlength\headsep{28.5pt}
  \setlength\topskip{6.9\p@}
  \setlength\footskip{11pt}
  \setlength\maxdepth{4\p@}
\fi

\ifnum\ios@journal@layout=5\relax
  \setlength\headsep{40\p@}%
\fi

% SIDE MARGINS
\setlength\oddsidemargin   {\@tempdima}
\setlength\evensidemargin  {\@tempdima}
\setlength\topmargin       {10mm}%
\ifcase\ios@journal@layout
\or
  \addtolength\topmargin{1.5mm}
\or
  \addtolength\topmargin{-5.9\p@}
\or
  \addtolength\topmargin{43\p@}
\or
  \addtolength\topmargin{5.06\p@}
\or
  \setlength\topmargin{-20\p@}%
\fi

% TEXT PARAMETERS
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@}

% BREAKS
\setlength\smallskipamount{6\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{12\p@ \@plus 2\p@ \@minus 3\p@}
\setlength\bigskipamount{24\p@ \@plus 6\p@ \@minus 6\p@}  

\ifcase\ios@journal@layout
\or
  \setlength\medskipamount{12\p@ \@plus 3\p@ \@minus 3\p@}
  \setlength\bigskipamount{24\p@ \@plus 3\p@ \@minus 3\p@}
\or
  \setlength\medskipamount{12\p@ \@plus 12\p@ \@minus 3\p@}
\else
  \setlength\medskipamount{12\p@ \@plus 3\p@ \@minus 3\p@}
  \setlength\bigskipamount{24\p@ \@plus 3\p@ \@minus 3\p@}
\fi

%% Page-breaking penalties
\ifnum\jobstart@date@info>20211020\relax
  \clubpenalty=4000
  \widowpenalty=4000
\else
  \clubpenalty=10000
  \widowpenalty=10000
\fi
\displaywidowpenalty=50
\ifnum\jobstart@date@info>20211020\relax
  \predisplaypenalty=10000
\fi
\ifnum\ios@journal@layout=\tw@
  \ifnum\jobstart@date@info>20211020\relax\else
    \clubpenalty=1000
    \widowpenalty=1000
    \displaywidowpenalty=0
    \predisplaypenalty=0
  \fi
\fi
\pretolerance=100
\ifnum\jobstart@date@info>20211020\relax
  \tolerance=1600
  \interdisplaylinepenalty=100
\else
  \tolerance=800
\fi
\hbadness=800
\ifnum\ios@journal@layout>\z@
  \emergencystretch=\z@
\else
  \ifnum\jobstart@date@info>20211020\relax
    \emergencystretch=\z@
  \else
    \emergencystretch=3\p@
  \fi
\fi
\hfuzz=1\p@

\ifnum\jobstart@date@info>20211020\relax
  \adjdemerits=10000
  \brokenpenalty=100
  \doublehyphendemerits=10000
\else
  \adjdemerits=0
  \brokenpenalty=0
  \doublehyphendemerits=0
\fi
\interlinepenalty=0

\if@twocolumn
  \setlength\marginparsep{10\p@}
  \ifnum\ios@journal@layout=\tw@
    \setlength\marginparsep{7\p@}
  \fi
\else
  \setlength\marginparsep{7\p@}
\fi
\setlength\marginparpush{5\p@}

\ifnum\ios@journal@layout=4
  \setlength\marginparwidth{.75in}
  \setlength\marginparsep{7\p@}
  \setlength\marginparpush{5\p@}
\fi

%% Footnotes
\setlength\footnotesep{6.65\p@}
\setlength{\skip\footins}{18\p@ \@plus 6\p@}

%% Floats
\setlength\floatsep       {15\p@ \@plus 10\p@ \@minus 4\p@}
\setlength\textfloatsep   {12\p@ \@plus 6\p@ \@minus 4\p@}
\setlength\intextsep      {12\p@ \@plus 6\p@ \@minus 4\p@}
\setlength\dblfloatsep    {15\p@ \@plus 10\p@ \@minus 4\p@}
\setlength\dbltextfloatsep{12\p@ \@plus 12\p@ \@minus 4\p@}

%%  For floats on a separate float page or column:
\setlength\@fptop{0\p@ \@plus 1fil}
\setlength\@fpsep{8\p@ \@plus 1000fil}
\setlength\@fpbot{0\p@ \@plus 1fil}
\setlength\@dblfptop{0\p@ \@plus 1fil}
\setlength\@dblfpsep{8\p@ \@plus 1000fil}
\setlength\@dblfpbot{0\p@ \@plus 1fil}

\setcounter{topnumber}{5}
\renewcommand\topfraction{.90}
\setcounter{bottomnumber}{5}
\renewcommand\bottomfraction{.90}
\setcounter{totalnumber}{10}
\renewcommand\textfraction{.10}
\renewcommand\floatpagefraction{.9}
\setcounter{dbltopnumber}{5}
\renewcommand\dbltopfraction{.99}
\renewcommand\dblfloatpagefraction{.8}

%% Penalties
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty

%% Lists
\setlength\partopsep{0\p@}
\def\@listI{%
  \leftmargin=\leftmargini
  \topsep=6pt plus 3pt minus 3pt
  \partopsep=3pt plus 1pt minus 2pt
  \itemsep=0pt plus 0.5pt minus 1pt
  \parsep=0pt
  }
\let\@listi\@listI
\@listi
\def\@listii{%
  \leftmargin=\leftmarginii
  \labelwidth=\leftmarginii
  \advance\labelwidth-\labelsep
  \topsep=4\p@ \@plus2\p@ \@minus\p@
  \parsep=0\p@ \@plus1\p@  \@minus\p@
  \itemsep=\parsep
  }
\def\@listiii{%
  \leftmargin=\leftmarginiii
  \labelwidth=\leftmarginiii
  \advance\labelwidth-\labelsep
  \topsep=2\p@ \@plus\p@\@minus\p@
  \parsep=\z@
  \partopsep=\p@ \@plus\z@ \@minus\p@
  \itemsep=\topsep
  }
\def\@listiv{%
  \leftmargin=\leftmarginiv
  \labelwidth=\leftmarginiv
  \advance\labelwidth-\labelsep
  }
\def\@listv{%
  \leftmargin=\leftmarginv
  \labelwidth=\leftmarginv
  \advance\labelwidth-\labelsep
  }
\def\@listvi{%
  \leftmargin=\leftmarginvi
  \labelwidth=\leftmarginvi
  \advance\labelwidth-\labelsep
  }

\DeclareMathSizes{\@xivpt}{\@xivpt}{\@xpt}{\@viiipt}
\DeclareMathSizes{12}{12}{\@viiipt}{\@viipt}

\csuse{\fmt@name @load@ps@headings}

% stubs for xelatex
\ifdefined\headerps@out\else
  \def\headerps@out#1{}%
\fi
\AtBeginDocument{%
  \ifdefined\pdf@box\else
    \newbox\pdf@box
  \fi
  \ifdefined\pdf@addtoksx\else
    \newtoks\pdf@addtoksx
  \fi
  }

%% empty RH
\def\ps@empty{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\hfill}%
  \def\@oddfoot{}%
  \let\@evenhead\@oddhead
  \let\@evenfoot\@oddfoot
  }
%% RH with pagenumber at bottom
\def\ps@plain{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\hfill}%
  \def\@oddfoot{}%
  \let\@evenhead\@oddhead
  \def\@oddfoot{\hfill\footnotesize\thepage\hfill}%
  \let\@evenfoot\@oddfoot
  }

\def\copyright@vskip{\vskip10pt}
\def\ps@copyright{%
  \let\@mkboth\@gobbletwo
  \def\@evenhead{\parbox[t]{\textwidth}{\ltitleheadline}}%
  \def\@oddhead{\parbox[t]{\textwidth}{\rtitleheadline}}%
  \def\@oddfoot{%
    \parindent\z@
    \parbox{\textwidth}{\copyright@vskip\copyright@size\@copyright}%
    }%
  \let\@evenfoot\@oddfoot
  }

\AtBeginDocument{%
  \@ifundefined{doc@stage}
    {\def\doc@stage{0}}
    {}%
  }
\let\rtitleheadline@size\scriptsize
\let\rtitleheadline@pagesize\footnotesize
\def\rtitleheadline{%
  \rtitleheadline@size
  \@journal\
  \ifnum\doc@stage<300\relax
    00 (20xx)%
  \else
    \@volume\ (\@pubyearfrom@fmt\@pubyear)%
  \fi
  \ \@pagerange
  \hfill
  \ifnum\ios@journal@layout=5\relax
    \raise -5\p@\hbox{\includegraphics[scale=.4]{sat1cim}}%
  \else
    {\rtitleheadline@pagesize\thepage}%
    \break
    \@ifundefined{@doi}{}{DOI \@doi\\}%
    \@publisher
  \fi
  }
\def\ltitleheadline{%
  {\rtitleheadline@pagesize\thepage}%
  \hfill
  \rtitleheadline@size
  \@journal\
  \ifnum\doc@stage<300\relax
    00 (20xx)%
  \else
    \@volume\ (\@pubyearfrom@fmt\@pubyear)%
  \fi
  \ \@pagerange
  \break
  \@ifundefined{@doi}{}{\null\hfill DOI \@doi\\}%
  \null\hfill\@publisher
  }

\def\lic@hyphen{}%
\AtBeginDocument{%
  \ifnum\ios@journal@layout=5
  \else
    \ifnum\ios@journal@layout=\thr@@
    \else
      \def\lic@hyphen{\char\hyphenchar\font\break}%
    \fi
  \fi
  %\ifnum\@pubyear=2020\relax
  %  \def\CC@BY@NC@license@text{This article is published online with Open Access and distributed under
  %    the terms of the Creative Commons Attribution Non-Commercial Li{\lic@hyphen}cense (CC BY-NC 4.0).}%
  %  \def\CC@BY@license@text{This article is published online with Open Access
  %    and distributed under the terms of the Creative Commons Attribution License (CC BY 4.0).}%
  %\else
    \def\CC@BY@NC@license@text{This is an Open Access article distributed under the terms of the
      Creative Commons Attribution-NonCommercial License (\href{https://creativecommons.org/licenses/by-nc/4.0/}{CC BY-NC 4.0}).}%
    \def\CC@BY@license@text{This is an Open Access article distributed under the terms of the
      Creative Commons Attribution License (\href{https://creativecommons.org/licenses/by/4.0/}{CC BY 4.0}).}%
  %\fi
  }%

\def\license@text{%
  \ifdefined\license@type
  \else
    \vtxClassError{\fmt@name}{\string\setpkgattr{license}{type}{} should be filled for Open Access articles. Possible Open Access license values: CC@BY@NC, CC@BY}{}%
  \fi
  \csname\license@type @license@text\endcsname
  }
\def\copyright@statement@tail{}%
\AtBeginDocument{%
  \ifdefined\HCode\else
    %\ifnum\@pubyear=2020\relax
    %\else
      \if@open@access
        \def\copyright@statement@tail{\ \license@text}%
      \fi
    %\fi
  \fi
  }%
\def\copyright@statement{%
  \copyright@sign\copyright@sign@delim
  \@pubyearfrom@fmt\@pubyear\@copyright@delim\@copyrightowner
  \copyright@statement@tail
  }
\def\standard@copyright@line{%
  \@issn\copyright@price\ %
  \copyright@statement
  }
\def\short@copyright@line{%
  \@issn\
  \copyright@statement
  }
\def\copyright@line{%
  \ifdefined\copyright@type
  \else
    \def\copyright@type{standard}%
  \fi
  \csname\copyright@type @copyright@line\endcsname
  }
\def\@copyright@license@prefix{}%
\def\@copyright@license@text{}%
\AtBeginDocument{%
  \expandafter\ifstrequal\expandafter{\@JID}{ICG}{%
    \def\@copyright@license@prefix{\vskip-2\baselineskip}%
    }{}%
  \expandafter\ifstrequal\expandafter{\@JID}{FRL}{%
    \def\@copyright@license@prefix{\vskip-2.5\baselineskip}%
    }{}%
  %\ifnum\@pubyear=2020\relax
  %  \if@open@access
  %    \def\@copyright@license@text{%
  %      \mbox{%
  %        \rlap{%
  %          \raisebox{-1.5\baselineskip}[\z@][\z@]{%
  %            \vtop{%
  %              \fontsize{7.9}\baselineskip\selectfont
  %              \license@text
  %              }%
  %            }%
  %          }%
  %        }%
  %      }%
  %  \fi
  %\fi
  }%
\def\@copyright{%
  \@copyright@license@prefix
  \@copyright@license@text
  \copyright@line
  }

%% Footnote
\def\footnoterule{%
  \kern-3\p@
  \hrule width\footnoterule@width
  \kern 2.6\p@
  }
\newcommand\@makefntext[1]{%
  \parindent1em
  \@makefnmark #1%
  }
\def\@makefnmark{%
  \@textsuperscript{\normalfont\@thefnmark}%
  }%
\def\freefootnotetext[#1]{%
  \begingroup
    \unrestored@protected@xdef\@thefnmark{#1}%
  \endgroup
  \@footnotetext
  }

%% Counters
\setcounter{secnumdepth}{3}
\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]
\renewcommand\thesection      {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection.\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}

%% Heading style
\def\section@style{\normalsize\bfseries\nohyphen\raggedright}
\def\subsection@style{\normalsize\itshape\nohyphen\raggedright}
\def\subsubsection@style{\normalsize\itshape\nohyphen\raggedright}
\def\paragraph@style{\normalsize\itshape}
\def\subparagraph@style{\normalsize\itshape}

\def\section@beforeskip{\bigskipamount}
\def\subsection@beforeskip{\medskipamount}
\def\subsubsection@beforeskip{\smallskipamount}
\def\paragraph@beforeskip{\smallskipamount}
\def\subparagraph@beforeskip{0.1pt}

\def\section@afterskip{\medskipamount}
\def\subsection@afterskip{\medskipamount}
\def\subsubsection@afterskip{0.1pt}
\def\paragraph@afterskip{-1em}
\def\subparagraph@afterskip{-1em}

\let\subsection@indent\z@
\let\subparagraph@indent\z@

\def\nohyphen{%
  \pretolerance=10000
  \tolerance=10000
  \hyphenpenalty=10000
  \exhyphenpenalty=10000
  }
\newcommand\section{%
  \@startsection{section}{1}{\z@}%
    {\section@beforeskip}%
    {\section@afterskip}%
    {\section@style}%
  }
\newcommand\subsection{%
  \@startsection{subsection}{2}{\subsection@indent}%
    {\subsection@beforeskip}%
    {\subsection@afterskip}%
    {\subsection@style}%
  }
\newcommand\subsubsection{%
  \@startsection{subsubsection}{3}{\z@}%
    {\subsubsection@beforeskip}%
    {\subsubsection@afterskip}%
    {\subsubsection@style}%
  }
\newcommand\paragraph{%
  \@startsection{paragraph}{4}{\z@}%
    {\paragraph@beforeskip}%
    {\paragraph@afterskip}%
    {\paragraph@style}%
  }
\newcommand\subparagraph{%
  \@startsection{subparagraph}{5}{\subparagraph@indent}%
    {\subparagraph@beforeskip}%
    {\subparagraph@afterskip}%
    {\subparagraph@style}%
  }
\def\@seccntformat#1{\csname the#1\endcsname.\enspace}
\let\app@sv@section\section

\def\@appsectionstar#1{%
  \if@seceqn
    \refstepcounter{section}%
  \fi
  \ifx&#1&%
    \app@sv@section*{Appendix}%
  \else
    \app@sv@section*{Appendix.\enspace #1}%
  \fi
  }
\def\@appsectionnostar#1{%
  \refstepcounter{section}%
  \ifx&#1&%
    \app@sv@section*{Appendix \thesection}%
  \else
    \app@sv@section*{Appendix \thesection.\enspace #1}%
  \fi
  }
\def\appendix{%
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \def\thesection{\Alph{section}}%
  \def\theHsection{\Alph{section}}%
  \def\section{\@ifstar\@appsectionstar\@appsectionnostar}%
  }

\def\acknowledgementsname{Acknowledgements}
\def\acknowledgements@error{%
  \vtxClassError{\fmt@name}{\string\ack(s) must be encoded as environment (for XML)}{}%
  }
\let\acknowledgements@closed\relax
\def\acknowledgements{%
  \global\let\acknowledgements@closed\acknowledgements@error
  \section*{\acknowledgementsname}%
  }
\def\endacknowledgements{%
  \global\let\acknowledgements@closed\relax
  }
\let\endack\endacknowledgements
\let\endacks\endacknowledgements
\AtEndDocument{\acknowledgements@closed}
\let\acks\acknowledgements
\def\ack{%
  \def\acknowledgementsname{Acknowledgement}%
  \acknowledgements
  }
\expandafter\ifstrequal\expandafter{\@JID}{JCS}
  {%
    \gdef\acknowledgementsname{Acknowledgments}%
    \gdef\ack{%
      \def\acknowledgementsname{Acknowledgment}%
      \acknowledgements
      }%
  }{}

%% Lists
\if@twocolumn
  \setlength\leftmargini{2em}
\else
  \setlength\leftmargini{2.5em}
\fi
\leftmargin=\leftmargini
\setlength\leftmarginii{2.2em}
\setlength\leftmarginiii{1.87em}
\setlength\leftmarginiv{1.7em}
\if@twocolumn
  \setlength\leftmarginv{.5em}
  \setlength\leftmarginvi{.5em}
\else
  \setlength\leftmarginv{1em}
  \setlength\leftmarginvi{1em}
\fi
\setlength\labelsep{.4em}
\setlength\labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}

\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}

\def\setenumlabel#1{\gdef\max@enumlabel{#1}}
\setenumlabel{1.}

\def\enumerate{%
  \@ifnextchar[%
    {\enumerate@}
    {\enumerate@[\max@enumlabel]}%
  }
\def\enumerate@[#1]{%
  \fix@thm@list
  \ifnum\@enumdepth>4\relax
    \@toodeep
  \else
    \advance\@enumdepth\@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
    \list{\csname label\@enumctr\endcsname}{%
      \usecounter{\@enumctr}%
      \def\makelabel##1{{\hfill\rm ##1}}%
      \settowidth{\labelwidth}{#1}%
      \advance\labelwidth by\parindent
      \labelsep=0.5em
      \leftmargin=\z@
      \rightmargin=\z@
      \itemindent=\labelwidth
      \advance\itemindent\labelsep
      \leftmargin=\the\itemindent
      \itemindent=\z@
      \partopsep=\z@
      \topsep=\smallskipamount
      \parsep=\z@
      \itemsep=\z@
      \listparindent=\z@
      }%
  \fi
  \setenumlabel{1.}%
  }

\RequirePackage{keyval}
\newcommand\labelitemi{\normalfont\bfseries\textendash}
\newcommand\labelitemii{\textasteriskcentered}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\let\@itemize@indent\parindent

\def\itemize{%
  \@ifnextchar[%
    {\itemize@}
    {\itemize@[]}%
  }
  
\def\itemize@[#1]{%
    \fix@thm@list
    \ifnum\@itemdepth>4\relax
      \@toodeep
    \else
      \advance\@itemdepth\@ne
      \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
      \if.#1.%
      \else
        \def\@@tempa{#1}%
        \edef\@itemitem{@@tempa}%
      \fi
      \list{\csname\@itemitem\endcsname}{%
        \settowidth{\labelwidth}%
        {\csname\@itemitem\endcsname}%
        \def\makelabel##1{##1}%
        \labelsep=0.5em
        \itemindent=\labelwidth
        \advance\itemindent\labelsep
        \advance\itemindent\@itemize@indent
        \leftmargin=\the\itemindent
        \itemindent=\z@
        \partopsep=\z@
        \topsep=\smallskipamount
        \parsep=\z@
        \itemsep=\z@
        \listparindent=\z@
        }%
    \fi
    }

\newenvironment{description}%
  {%
    \fix@thm@list
    \list{}{%
      \labelwidth=\z@
      \itemindent=-\leftmargin
      \let\makelabel\descriptionlabel
      }%
  }{%
    \endlist
  }
\newcommand*\descriptionlabel[1]{%
  \hspace\labelsep
  \normalfont\bfseries #1%
  }
\newenvironment{verse}
  {%
   \let\\\@centercr
   \list{}{%
     \itemsep=\z@
     \itemindent=-1.5em
     \listparindent=\itemindent
     \rightmargin=\leftmargin
     \advance\leftmargin 1.5em
     }%
   \item\relax
  }{%
    \endlist
  }

\newenvironment{quotation}
  {%
    \ifhmode
      \endgraf
    \fi
    \list{}{%
      \small
      \listparindent=2mm
      \itemindent=\z@
      \rightmargin=\z@
      \leftmargin=\parindent
      \partopsep=\z@
      \topsep=\smallskipamount
      \parsep=\z@
      }%
    \item[\Q@strut]\relax
  }{%
    \endlist
  }
\def\Q@strut{\leavevmode\hbox{\vrule height9pt depth1pt width0pt}}

\newenvironment{quote}
  {%
    \list{}{%
      \listparindent=\z@
      \itemindent=\listparindent
      \rightmargin=\z@
      \leftmargin=1.5em
      \partopsep=\z@
      \topsep=6\p@
      \parsep=\z@
      }%
    \item[\Q@strut]\relax
  }{%
    \endlist
  }

\def\abbrsection#1{%
  \edef\subsection@indent{-\leftmargin}%
  \subsection*{#1}%
  }
\define@key{abbrgroup}{title}{\def\ios@abbrgroup@title{#1}}
\define@key{abbrgroup}{notitle}[true]{\edef\ios@abbrgroup@notitle{1}}
\def\abbrgroup@labelwidth{0pt}
\newenvironment{abbrgroup}[1][title={Abbreviations}]%
  {%
    \setkeys{abbrgroup}{#1}%
    \def\abbrgroup@@labelwidth{0pt}%
    \@ifundefined{ios@abbrgroup@notitle}
      {\section*{\ios@abbrgroup@title}}
      {}%
    \list{}{%
      \labelwidth=\abbrgroup@labelwidth 
      \labelsep=1em%
      \leftmargin=\labelwidth
      \advance\leftmargin\labelsep
      \let\makelabel\abbrgrouplabel
      }%
  }{%
    \protected@write\@auxout{}{\string\gdef\string\abbrgroup@labelwidth{\abbrgroup@@labelwidth}}%
    \endlist
  }
\newcommand*\abbrgrouplabel[1]{%
  \normalfont
  \settowidth\@tempdima{#1}%
  \ifdim\@tempdima>\abbrgroup@@labelwidth
    \xdef\abbrgroup@@labelwidth{\the\@tempdima}%
  \fi
  #1%
  \hfill
  }

%% list label correction
\ifnum\ios@journal@layout>\z@
  \renewcommand\labelitemi{\normalfont$\bullet$}
  \renewcommand\labelenumi{(\theenumi)}
\fi

\csuse{\fmt@name @load@stabular}
\csuse{\fmt@name @load@tmultirow}

%% tabular
\let\savehline\hline
\def\thline{%
  \noalign{\vskip3pt}%
  \savehline
  \noalign{\vskip3pt}%
  }%
\def\fhline{%
  \noalign{\vskip1pt}%
  \savehline
  \noalign{\vskip7pt}%
  }%
\def\bhline{%
  \noalign{\vskip3pt}%
  \noalign{\global\arrayrulewidth=1\p@}%
  \savehline
  \noalign{\global\arrayrulewidth=.5\p@}%
  \noalign{\vskip3pt}%
  }%
\def\lhline{%
  \noalign{\vskip3pt}%
  \noalign{\global\arrayrulewidth=.3\p@}%
  \savehline
  \noalign{\global\arrayrulewidth=.5\p@}%
  \noalign{\vskip3pt}%
  }

%% Math settings
\setlength\mathindent{2em}
\setlength\arraycolsep{1.2\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\setlength\jot{6\p@}
\skip\@mpfootins=\skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}

\if@seceqn
  \@addtoreset{equation}{section}
  \renewcommand\theequation{\thesection.\@arabic\c@equation}
\else
  \renewcommand\theequation{\@arabic\c@equation}
\fi


\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\if@reducehelvetica
  \RequirePackage[scaled=.9]{helvet}
\else
  \RequirePackage{helvet}
\fi
\csuse{\fmt@name @load@textcomp}
\normalfont

\ifnum\ios@journal@layout=5\relax
  \RequirePackage{lmodern}%
\else
  \csuse{\fmt@name @load@mathtime}%
\fi

\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\allowdisplaybreaks

%% Tarp teoremos ir listo reikia ideti \quitvmode, kad 
%% itemas prasidedu is naujos eilutes. Aplinku sekima darome per \penalty 1234
\def\vtx@theorem@marker{1234}
\def\vtx@theorem@penalty{\penalty\vtx@theorem@marker}
\def\fix@thm@list{%
  \ifnum\the\lastpenalty=\vtx@theorem@marker\relax
    \quitvmode
  \fi
  }

\def\@begintheorem#1#2[#3]{%
  \deferred@thm@head{\the\thm@headfont \thm@indent
    \@ifempty{#1}{\let\thmname\@gobble}{\let\thmname\@iden}%
    \@ifempty{#2}{\let\thmnumber\@gobble}{\let\thmnumber\@iden}%
    \@ifempty{#3}{\let\thmnote\@gobble}{\let\thmnote\@iden}%
    \thm@swap\swappedhead\thmhead{#1}{#2}{#3}%
    \the\thm@headpunct
    \thmheadnl % possibly a newline.
    \hskip\thm@headsep
  }%
  \vtx@theorem@penalty
  \ignorespaces}

% gather fix to raise error on \label + \nonumber in one row
\def\math@cr@@@gather{%
  \ifst@rred\nonumber\fi
  &\if@eqnsw \global\tag@true \fi\relax
  \iftag@
      \make@display@tag
  \fi
  \ifst@rred\else\global\@eqnswtrue\fi
  \global\advance\row@\@ne
  \cr
  }

\csuse{\fmt@name @load@alt@mathtime}

\ifnum\ios@journal@layout>\z@
  %% different from mathtime settings only
  \DeclareMathSizes{9}{9}{7}{6}
  \DeclareMathSizes{11}{11}{7.6}{6}
  \DeclareMathSizes{12}{12}{8}{7}
  \DeclareMathSizes{14}{14}{10}{8}
  \DeclareMathSizes{16}{16}{12}{9}
  \DeclareMathSizes{17}{17}{12}{9}
  \DeclareMathSizes{20}{20}{14}{10}
  \DeclareMathSizes{24}{24}{16}{12}
  \DeclareMathSizes{25}{25}{16}{12}
\fi


\def\amsarraycolsep{\arraycolsep=5\p@}
%% command for changing arraycolsep:
\def\setamsmatrix{%
  \let\sv@matrix\matrix
  \def\matrix{\amsarraycolsep\sv@matrix}% 
  \let\sv@pmatrix\pmatrix
  \def\pmatrix{\amsarraycolsep\sv@pmatrix}% 
  \let\sv@bmatrix\bmatrix
  \def\bmatrix{\amsarraycolsep\sv@bmatrix}% 
  \let\sv@Bmatrix\Bmatrix
  \def\Bmatrix{\amsarraycolsep\sv@Bmatrix}%
  \let\sv@vmatrix\vmatrix
  \def\vmatrix{\amsarraycolsep\sv@vmatrix}%
  \let\sv@Vmatrix\Vmatrix
  \def\Vmatrix{\amsarraycolsep\sv@Vmatrix}%
  }
\setamsmatrix

\ifnum\ios@journal@layout>\z@
  \csuse{\fmt@name @load@timesxi}
\fi
\csuse{\fmt@name @vtex@gr@packages}

%% Last page control
\csuse{\fmt@name @load@vtex@checklpage}

%% Theorems
\ifnum\ios@journal@layout>\z@
  \def\mathend@qed{%
    \hbox to\hsize{%
      \hfill$\Box$%
      }%
    }%
  \def\upqed{%
    \vskip-\belowdisplayskip
    \vskip-\baselineskip
    \kern0pt
    \let\qed\mathend@qed
    }%
  \def\qed{%
    \ifmmode
      \hskip2em\Box
    \else
      \leavevmode
      \unskip\penalty9999
      \hbox{}\nobreak\quad $\Box$%
    \fi
    }%
  \def\noqed{%
    \let\sv@qed\qed
    \def\qed{%
      \global\let\qed\sv@qed
      }%
    }%
  \def\@proof@label@sep{.\kern\p@}
  \def\proof@topsep{9\p@\@plus3\p@}
  \renewenvironment{proof}[1][\proofname]%
    {%
      \par
      \pushQED{\qed}%
      \normalfont
      \topsep=\proof@topsep
      \relax
      \trivlist
      \item[%
        \hskip\labelsep
        \bfseries
        #1\@proof@label@sep]%
      \ignorespaces
    }{%
      \popQED
      \endtrivlist
      \@endpetrue
    }
    \newtheoremstyle{plain}
      {\medskipamount}{\medskipamount}{\normalfont\itshape}{}%
      {\normalfont\bfseries}{.}{.5em}{}%
  \newtheoremstyle{remark}
    {\medskipamount}{\medskipamount}{\normalfont\upshape}{}%
    {\normalfont\bfseries}{.}{.5em}{}%
  \g@addto@macro\@endtheorem{\@endpetrue}%
  \g@addto@macro\endproof{\@endpetrue}%
\fi

\g@addto@macro\@endtheorem{\@endpefalse}%
\g@addto@macro\endproof{\@endpefalse}%

%% Floats
\newcounter{figure}
\if@secfloat
  \@addtoreset{figure}{section}
  \renewcommand\thefigure{\thesection.\@arabic\c@figure}
\else
  \renewcommand\thefigure{\@arabic\c@figure}
\fi
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure.}
\newenvironment{figure}
  {%
    \let\@makecaption\@makefigurecaption
    \let\@floatboxreset\@figureboxreset
    \@float{figure}%
  }{%
    \end@float
  }
\newenvironment{figure*}
  {%
    \let\@makecaption\@makefigurecaption
    \let\@floatboxreset\@figureboxreset
    \@dblfloat{figure}%
  }{%
    \end@dblfloat
  }
\def\@figureboxreset{%
  \reset@font
  \centering
  \@setnobreak
  \@setminipage
  }
\long\def\@makefigurecaption#1#2{%
  \footnotesize
  \vskip\abovecaptionskip
  \setbox\@tempboxa\hbox{#1 #2}%
  \ifdim\wd\@tempboxa>\hsize
    \unhbox\@tempboxa
    \par
  \else
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  }

\newcounter{listing}
\def\listingname{Listing}
\renewcommand\thelisting{\@arabic\c@listing}
\def\fps@listing{tbp}
\def\ftype@listing{5}
\def\ext@listing{lol}
\def\fnum@listing{\listingname~\thelisting.}
\newenvironment{listing}
  {%
    \let\@makecaption\@makefigurecaption
    \let\@floatboxreset\@figureboxreset
    \@float{listing}%
  }{%
    \end@float
  }
\newenvironment{listing*}
  {%
    \let\@makecaption\@makefigurecaption
    \let\@floatboxreset\@figureboxreset
    \@dblfloat{listing}%
  }{%
    \end@dblfloat
  }

\newcounter{table}
\if@secfloat
  \@addtoreset{table}{section}
  \renewcommand\thetable{\thesection.\@arabic\c@table}
\else
  \renewcommand\thetable{\@arabic\c@table}
\fi
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\let\tablebr\break
\newenvironment{table}
  {%
    \let\@makecaption\@maketablecaption
    \let\@floatboxreset\@tableboxreset
    \@float{table}%
  }{%
    \end@float
  }
\newenvironment{table*}
  {%
    \tablewidth=\textwidth
    \let\@makecaption\@maketablecaption%
    \let\@floatboxreset\@tableboxreset
    \@dblfloat{table}%
  }{%
    \end@dblfloat
  }
\def\@tableboxreset{%
  \reset@font
  \centering\tablefont
  \def\arraystretch{1.2}%
  \@setnobreak
  \@setminipage
  }

\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{8\p@}
\setlength\belowcaptionskip{3\p@}

\newdimen\tablewidth
\tablewidth\textwidth
\long\def\@maketablecaption#1#2{%
  \begingroup
    \footnotesize
    \global\setbox\@tempboxa\hbox{#2}%
  \endgroup
  \centering
  \hbox to \hsize{\footnotesize\hss#1\hss}%
  \par
  \ifdim\wd\@tempboxa>\tablewidth
    \parbox[t]{\tablewidth}{%
      \footnotesize
      \mbox{}%
      #2%
      \vphantom{Ay}%
      \par
      }%
  \else
    \hbox to\hsize{\hfill\box\@tempboxa\vphantom{Ay}\hfill}%
  \fi
  \vskip\belowcaptionskip
  }

\csuse{\fmt@name @load@tabnotes}

\if@rotating@loaded
  \newenvironment{local@sidewaysfigure}
    {%
      \let\@makecaption\@makefigurecaption
      \let\@floatboxreset\@figureboxreset
      \@rotfloat{figure}%
    }{%
      \end@rotfloat
    }
  \newenvironment{local@sidewaysfigure@star}
    {%
      \let\@makecaption\@makefigurecaption
      \let\@floatboxreset\@figureboxreset
      \@rotdblfloat{figure}%
    }{%
      \end@rotdblfloat
    }
  \newenvironment{local@sidewaystable}
    {%
      \tablewidth=\textheight
      \let\@makecaption\@maketablecaption
      \let\@floatboxreset\@tableboxreset
      \@rotfloat{table}%
    }{%
      \end@rotfloat
    }
  \newenvironment{local@sidewaystable@star}
    {%
      \tablewidth=\textheight
      \let\@makecaption\@maketablecaption
      \let\@floatboxreset\@tableboxreset
      \@rotdblfloat{table}%
    }{%
      \end@rotdblfloat
    }
  \AtEndOfClass{%
    \def\sidewaysfigure{\local@sidewaysfigure}%
    \def\endsidewaysfigure{\endlocal@sidewaysfigure}%
    \@namedef{sidewaysfigure*}{\local@sidewaysfigure@star}%
    \@namedef{endsidewaysfigure*}{\endlocal@sidewaysfigure@star}%
    \def\sidewaystable{\local@sidewaystable}%
    \def\endsidewaystable{\endlocal@sidewaystable}%
    \@namedef{sidewaystable*}{\local@sidewaystable@star}%
    \@namedef{endsidewaystable*}{\endlocal@sidewaystable@star}%
    }
\fi

%% Compatibility with old latex
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\let\sl\it
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}

\if@secthm
  \@addtoreset{thm}{section}
  \def\thethm{\thesection.\arabic{thm}}
\else
  \def\thethm{\arabic{thm}}
\fi

%% Bibliography
\newenvironment{thebibliography}
  {}{}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\def\@citex[#1]#2{%
  \let\@citea\@empty
  \@cite{%
    \@for\@citeb:=#2\do{%
      \@citea
      \def\@citea{,\penalty\@m\hskip.1pt}%
      \edef\@citeb{\expandafter\@firstofone\@citeb}%
      \if@filesw
        \immediate\write\@auxout{\string\citation{\@citeb}}%
      \fi
      \@ifundefined{b@\@citeb}
        {%
          \mbox{\reset@font\bfseries ?}%
          \G@refundefinedtrue
          \ClassWarning{iosart2x}{Citation `\@citeb' on page \thepage \space undefined}%
        }{%
          \hbox{\csname b@\@citeb\endcsname}%
        }%
      }%
    }%
  {#1}%
  }

\if@nameyear
  \csuse{\fmt@name @natbib@authoryear}
\else
  \csuse{\fmt@name @natbib@numbers}
\fi

\csuse{\fmt@name @natbib@setup}
\long\def\bibfont{%
  \footnotesize
  \baselineskip=\dimexpr1\baselineskip\@plus.5\p@\relax
  }
\bibsep=\z@\@plus.5\p@
\let\endbibitem\relax

\RequirePackage{color}
\csuse{\fmt@name @qtags@and@etc}
\csuse{\fmt@name @vtex@ltxext}
\def\getattribute#1#2{\csname #1@#2\endcsname}
\def\sep@key@value#1=#2/?/#3{\setpkgattr{#3}{#1}{#2}}
\def\do@option@list#1#2{%
  \@for\curr@option:={#2}\do{%
    \expandafter\sep@key@value\curr@option/?/{#1}\relax
    }%
  }

\RequirePackage{url}
\csuse{\fmt@name @vtex@url}
\csuse{\fmt@name @define@url@macro}
\urlstyle{rm}

% frl200019
\AtBeginDocument{%
  \let\url@sav\url
  \def\url#1{%
    \def\${\string$}%
    \url@sav{#1}%
    }%
  \let\surl@sav\surl
  \def\surl#1{%
    \def\${\string$}%
    \surl@sav{#1}%
    }%
  }%

\@ifundefined{tagbib@style}
  {\def\tagbib@style{}}
  {}
\csuse{\fmt@name @bibl@settings}

%% history month - full name
\def\setvaluelist#1#2{%
  \@tempcnta=0\relax
  \@for\@curr@val:=#2\do{%
    \advance\@tempcnta by1\relax
    \expandafter\protected@xdef\csname #1@item@\the\@tempcnta\endcsname{\@curr@val}%
    }%
  \expandafter\protected@xdef\csname #1@item@0\endcsname{\the\@tempcnta}%
  }
\xdef\getitemvalue#1#2{\noexpand\csname #1@item@#2\endcsname}
\def\sday#1{#1}
\def\syear#1{#1}
\def\smonth#1{%
  \@ifundefined{month@item@#1}%
    {\vtxClassError{\fmt@name}{There is no month with number '#1'!}{}}
    {\getitemvalue{month}{#1}}%
  }%
\setvaluelist{month}{January,February,March,April,May,June,July,August,September,October,November,December}
\expandafter\def\csname month@item@0\endcsname{????}

\newif\if@history \@historyfalse

\def\accepted#1{%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@accepted{#1}%
    \global\@historytrue
  \fi
  }
\def\@accepted{\@nil}
\def\communicated#1{%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@communicated{#1}%
    \global\@historytrue
  \fi
  }
\def\@communicated{\@nil}
\def\dedicated#1{%
  \expandafter\ifstrequal\expandafter{\@JID}{SAT}{%
    \vtxClassError{\fmt@name}{Style is not prepared for \string\dedicated}{}%
    }%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@dedicated{#1}%
    \global\@historytrue
  \fi
  }
\def\@dedicated{\@nil}
\def\published#1{%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@published{#1}%
    \global\@historytrue
  \fi
  }
\def\@published{\@nil}
\def\received#1{%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@received{#1}%
    \global\@historytrue
  \fi
  }
\def\@received{\@nil}
\def\revised#1{%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@revised{#1}%
    \global\@historytrue
  \fi
  }
\def\@revised{\@nil}
\def\submitted#1{%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@submitted{#1}%
    \global\@historytrue
  \fi
  }
\def\@submitted{\@nil}

\def\acceptedby#1{\gdef\@acceptedby{#1}}

\def\accepted@prefix{Accepted~}
\def\acceptedby@prefix{Accepted~by:~}
\def\communicated@prefix{Communicated~by~}
\def\published@prefix{Published~}
\def\received@prefix{Received~}
\def\revised@prefix{Revised~}
\def\submitted@prefix{Submitted~}

\def\dedicated@prefix{%
  \bgroup
  \vskip\b@history@skip
  \leavevmode
  \normalsize\itshape
  }

\def\history@prefix{}

\def\empty@data{\@nil}
\def\history@fmt{%
  \bgroup
    \history@size
    \vskip\b@history@skip
    \history@prefix
    \leavevmode
    \ifnum\ios@journal@layout=5\relax
      \hsize=\abstract@width
      \keywords@indent
      \hskip\abstract@indent@val
    \fi
    %
    \ifx\@received\empty@data
    \else        
      \received@prefix\@received\received@postfix
      \upd@hist@prefices
    \fi
    \ifx\@submitted\empty@data
    \else        
      \submitted@prefix\@submitted\submitted@postfix
      \upd@hist@prefices
    \fi
    \ifx\@revised\empty@data
    \else
      \revised@prefix\@revised\revised@postfix
      \upd@hist@prefices
    \fi
    \ifx\@accepted\empty@data
    \else
      \accepted@prefix\@accepted\accepted@postfix
      \upd@hist@prefices
    \fi
    \ifx\@communicated\empty@data
    \else
      \communicated@prefix\@communicated\communicated@postfix
      \upd@hist@prefices
    \fi
    \ifx\@published\empty@data
    \else
      \published@prefix\@published\published@postfix
      \upd@hist@prefices
    \fi
    \history@postfix
    \ifx\@dedicated\empty@data
    \else
      \dedicated@prefix\@dedicated\dedicated@postfix
      \upd@hist@prefices
    \fi
    %
    \par
  \egroup
  }

%% >>> Frontmatter >>>
\newtoks\t@glob@notes
\newtoks\t@loc@notes
\newcount\note@cnt
\newcounter{author}
\newcount\n@authors
\newcounter{address}
\newdimen\sv@mathsurround
\newcount\sv@hyphenpenalty
\newcount\prev@elem \prev@elem=0
\newcount\cur@elem  \cur@elem=0
\chardef\e@pretitle=1
\chardef\e@title=1
\chardef\e@subtitle=1
\chardef\e@author=2
\chardef\e@address=3

\newif\if@newelem
\newif\if@firstauthor
\newif\if@preface
\newif\if@hasabstract
\newif\if@haskeywords

\newbox\fm@box
\newdimen\fm@size
\newbox\t@abstract
\newbox\t@keywords

\def\add@tok#1#2{%
  \global#1\expandafter{\the#1#2}%
  }
\def\add@xtok#1#2{%
  \begingroup
    \no@harm
    \xdef\@act{\global\noexpand#1{\the#1#2}}%
    \@act
  \endgroup
  }
\def\tailthanksref[#1]#2{\noexpand\pthanksref{#1}}
\def\pthanksref#1{%
  \global\advance\note@cnt\@ne
  \ifnum\note@cnt>\@ne
    \global\t@loc@notes\expandafter{\the\t@loc@notes\note@sep}%
  \fi
  \global\t@loc@notes\expandafter{\the\t@loc@notes#1}%
  }
\def\beg@elem{%
  \global\t@loc@notes={}%
  \global\note@cnt\z@
  }
\def\@xnamedef#1{\expandafter\xdef\csname #1\endcsname}
\def\no@harm{%
  \let\\=\relax
  \let\rm\relax
  \let\ss=\relax
  \let\ae=\relax
  \let\oe=\relax
  \let\AE=\relax
  \let\OE=\relax
  \let\o=\relax
  \let\O=\relax
  \let\i=\relax
  \let\j=\relax
  \let\aa=\relax
  \let\AA=\relax
  \let\l=\relax
  \let\L=\relax
  \let\d=\relax
  \let\b=\relax
  \let\c=\relax
  \let\bar=\relax
  \def\qq@group@start{}%
  \def\qq@group@end{}%
  \let\qq@cmd\@firstofone
  \def\qq@style{}%
  \def\protect{\noexpand\protect\noexpand}%
  }
\def\upd@elem#1{%
  \prev@elem=\cur@elem
  \cur@elem=\csname e@#1\endcsname
  }
\def\proc@elem#1#2{%
  \begingroup
    \no@harm
    \def\thanks##1##{\@gobble}%
    \def\thanksref##1##{\@gobble}%
    \@xnamedef{@#1}{#2}%
  \endgroup
  \upd@elem{#1}%
  \expandafter\elem@nothanksref#2\thanksref\relax
  \expandafter\elem@nothanks#2\thanks\relax
  }

\def\elem@nothanksref#1\thanksref{\futurelet\@peektok\elem@thanksref}
\def\elem@thanksref{%
  \ifx\@peektok\relax
  \else
    \expandafter\elem@morethanksref
  \fi
  }
\def\elem@morethanksref[#1]#2{%
  \add@thanks{#1}%
  \elem@nothanksref
  }
\def\elem@nothanks#1\thanks{\futurelet\@peektok\elem@thanks}
\def\elem@thanks{%
  \ifx\@peektok\relax
  \else
    \ifx\@peektok[%
      \expandafter\expandafter\expandafter\elem@morethankse
    \else
      \expandafter\expandafter\expandafter\elem@morethanks
    \fi
  \fi
  }
\def\elem@morethankse[#1]#2{%
  \thanks@optarg[#1]{#2}%
  \add@thanks{#1}%
  \elem@nothanks
  }
\def\elem@morethanks#1{%
  \thanks@optarg[]{#1}%
  \add@thanks{}%
  \elem@nothanks
  }

\def\add@thanks#1{%
  \global\advance\note@cnt\@ne
  \ifnum\note@cnt>\@ne
    \add@xtok\t@loc@notes{\note@sep}%
  \fi
  \ifx.#1.%
    \add@xtok\t@loc@notes{%
      \protect\Hy@breaklinkstrue
      \protect\hyper@linkstart{link}{\thefootnote iosthanks}%
      \thefootnote
      \protect\hyper@linkend
      }%
  \else
    \add@xtok\t@loc@notes{%
      \protect\Hy@breaklinkstrue
      \protect\hyper@linkstart{link}{#1iosthanks}%
      #1%
      \protect\hyper@linkend
      }%
  \fi
  }

\def\add@addressref#1{%
  \ifnum\ios@journal@layout=5\else
    \global\advance\note@cnt\@ne
    \ifnum\note@cnt>\@ne
      \add@xtok\t@loc@notes{\note@sep}%
    \fi
    \add@tok\t@loc@notes{\ref{#1}}%
  \fi
  }
\def\note@sep{,}

\let\glob@notes@out\@firstofone
\def\thanks@optarg[#1]#2{%
  \ifx.#1.%
    \add@tok\t@glob@notes{\footnotetext}%
  \else
    \add@tok\t@glob@notes{\freefootnotetext}%
  \fi
  \ifx.#1.%
    \refstepcounter{footnote}%
    \add@xtok\t@glob@notes{[\the\c@footnote]}%
  \else
    \add@xtok\t@glob@notes{[#1]}%
  \fi
  \ifx.#1.%
    \edef\my@hy@tmp{\noexpand\hypertarget{\thefootnote iosthanks}{}}%
  \else
    \edef\my@hy@tmp{\noexpand\hypertarget{#1iosthanks}{}}%
  \fi
  \expandafter\add@tok\expandafter\t@glob@notes\expandafter{\expandafter{\my@hy@tmp
    \if@nameyear
    \else
      \@if@vtex@in@house{%
        \def\abstract@cite@prompt{\string\thanks}%
        \let\cite\abstract@cite
        }{}%
    \fi
    \glob@notes@out{#2}}}%
  \ignorespaces
  }
  
\def\check@for@email#1\printead#2\printead{%
  \if.#2.%
  \else
    \raggedright
  \fi
  }%
\def\do#1{%
  \def\par@jid{#1}%
  \ifx\@jid\par@jid
    \def\glob@notes@out##1{%
      \setbox\@tempboxa\hbox{##1}%
      \@tempdima=\dimexpr\parindent+\wd\@tempboxa+3\p@\relax
      \ifdim\@tempdima>\columnwidth\relax
        \let\printead@sav\printead
        \let\printead\relax
        \check@for@email##1\printead\printead
        \let\printead\printead@sav
      \fi
      ##1%
      }%
  \fi
  }
\docsvlist{ais,aic,sw,web,scs}%

\def\fm@textsuperscript#1{%
  \textsuperscript{\upshape #1}%
  }
\ifnum\ios@journal@layout>\z@
  \def\fm@textsuperscript#1{\hbox{$^{\mathrm{\scriptsize #1}}$}}
\fi

\newdimen\a@title@skip \a@title@skip=12\p@
\newskip\b@section@skip \b@section@skip=36\p@ plus6\p@ minus6\p@%
\newskip\b@pretitle@skip \b@pretitle@skip12\p@

\def\frontmatter{%
  \def\pdftitle##1{\write@pdfinfo{\user@hy@title}{##1}}%
  \def\pdfauthor##1{\write@pdfinfo{\user@hy@author}{##1}}%
  \def\pdfsubject##1{\write@pdfinfo{\user@hy@subject}{##1}}%
  \def\pdfkeywords##1{\write@pdfinfo{\user@hy@keywords}{##1}}%
  \def\dead{\hbox{$\dagger$}}%
  \let\@corresp@note\relax
  \global\t@glob@notes={}%
  \global\c@author=\z@
  \global\c@address=\z@
  \sv@mathsurround=\mathsurround\m@th
  % \global\n@authors=0\the\n@authors\relax
  % \global\advance\n@authors\m@ne
  \global\@firstauthortrue
  \global\@hasabstractfalse
  \global\@prefacefalse
  \newcount\c@sv@footnote
  \global\c@sv@footnote=\c@footnote
  \parindent=\z@
  \open@fm
  \ignorespaces
  }
\def\preface{\@prefacetrue}

\def\endfrontmatter{%
  \global\c@footnote=\c@sv@footnote
  % \global\n@authors=\the\c@author
  % \@writecount
  \csname form@runauthors\endcsname
  \@if@vtex@in@house
    {%
      \@ifundefined{pretitle@used}
        {\vtxClassError{\fmt@name}{Command \string\pretitle[XML]{...} must be used}{}}
        {}%
    }{}%
  \global\@topnum=\z@
  \ifx\@firstpage\@lastpage
    \gdef\@pagerange{\special@supplement@letter\thefirstpage}%
  \else
    \gdef\@pagerange{\special@supplement@letter\thefirstpage--\special@supplement@letter\thelastpage}%
  \fi
  \thispagestyle{copyright}%
  \ifnum\ios@journal@layout=5\relax
    \if@make@lua@auth@
      \vtx@directlua{%
        make_authors()
        }%
    \fi
  \else
    \if@preface
    \else
      \if@history
        \history@fmt
      \fi
    \fi
  \fi
  \newcount\c@sv@footnote
  \global\c@sv@footnote=\c@footnote
  \if@twocolumn
  \else
    \output@glob@notes
  \fi
  \if@preface
    \@hasabstractfalse
  \fi
  \if@hasabstract
    \normal@text
    \vskip\abstract@skip
    \unvbox\t@abstract
  \fi
  \if@haskeywords
    \ifnum\jobstart@date@info>20211001\relax\else
      \if@hasabstract
        \vskip\keywords@skip
      \else
        \vskip\keywords@noabs@skip
      \fi
    \fi
    \unvbox\t@keywords
  \fi
  \@ifundefined{@acceptedby}
    {}
    {%
      \ifnum\prevdepth>\z@
        \vskip-\prevdepth
      \fi
      \vskip\keywords@skip
      \bgroup
        \abstract@size
        \acceptedby@prefix
        \@acceptedby
        \acceptedby@postfix
      \egroup
    }%
  \ifnum\ios@journal@layout=5\relax
    \if@history
      \history@fmt
    \fi
  \fi
  \@ifundefined{has@notes}
    {}
    {%
      \vskip\notes@skip
      \unvbox\notes@box
      \notes@end@hook
    }%
  \@tempcnta\z@
  \@whilenum\@tempcnta<\c@author\do{%
    \advance\@tempcnta\@ne
    \ifnum\@tempcnta>\@ne
      \addto@authors@list{, }%
    \fi
    \addto@authors@list{%
      \@ifundefined{\short@journal @fnmsmd@\the\@tempcnta @author}
        {}
        {\@nameuse{\short@journal @fnmsmd@\the\@tempcnta @author} }%
      \@nameuse{\short@journal @snm@\the\@tempcnta @author}%
      }%
    }%
  \write@pdfinfo{\hy@author}{\the\authors@list}%
  \write@pdfinfo{\hy@subject}{\@journal, \@pubyearfrom@fmt\@pubyear, \@volume, \@issue, \@pagerange}%
  \write@pdfinfo{\hy@keywords}{\the\keywords@list}%
  \close@fm
  \if@twocolumn
    \output@glob@notes
  \fi
  \markboth{\@runauthor\@runtitle}{\@runauthor\@runtitle}%
  \global\@prefacefalse
  \global\leftskip=\z@
  \global\@rightskip=\z@
  \global\rightskip=\@rightskip
  \global\c@footnote=\c@sv@footnote
  \global\mathsurround=\sv@mathsurround
  \let\title\relax
  \let\author\relax
  \let\address\relax
  \let\frontmatter\relax
  \let\endfrontmatter\relax
  \let\@maketitle\relax
  \let\@@maketitle\relax
  \normal@text
  }
%% Twocolumn journal has full page width commands
%% \pretitle, \title and \subtitle only. So we're using
%% \maketitle, which closes box
\def\two@c@maketitle{%
    \global\let\close@fm\relax
    \vskip\b@section@skip
    \par
  \egroup
  \ifnum\jobstart@date@info>20211020\relax\else
    \emergencystretch=1pc
  \fi
  \twocolumn[\unvbox\fm@box]%
  }

\if@restonecol
  \let\maketitle\relax
\else
  \ifnum\ios@journal@layout>\z@
    \let\maketitle\@empty
  \else
    \let\maketitle\two@c@maketitle
  \fi
\fi

\newdimen\t@xtheight
\def\init@settings{%
  \splittopskip=\topskip
  \splitmaxdepth=\maxdepth
  \t@xtheight=\textheight
  \advance\t@xtheight-\splittopskip
  }
\def\open@fm{%
  \global\setbox\fm@box=\vbox\bgroup
    \hsize=\textwidth
    \raggedright
    \sv@hyphenpenalty=\hyphenpenalty
    \hyphenpenalty=\@M
  }

\def\close@fm{%
    \vskip\b@section@skip
    \par
  \egroup
  \fm@size=\dp\fm@box
  \advance\fm@size by \ht\fm@box
  \@whiledim\fm@size>\t@xtheight\do{%
    \global\setbox\@tempboxa=\vsplit\fm@box to \t@xtheight
    \unvbox\@tempboxa \clearpage
    \fm@size=\dp\fm@box \advance\fm@size by \ht\fm@box
    }
  \if@twocolumn
    \ifnum\jobstart@date@info>20211020\relax\else
      \emergencystretch=1pc
    \fi
    \twocolumn[\unvbox\fm@box]%
  \else
    \unvbox\fm@box
  \fi
  }

\def\output@glob@notes{%
  \bgroup
    \the\t@glob@notes
  \egroup
  }

\def\justify@off{%
  \let\\=\@normalcr
  \leftskip=\z@
  \@rightskip=\@flushglue
  \rightskip=\@rightskip
  }
\def\justify@on{%
  \let\\=\@normalcr
  \leftskip=\z@
  \@rightskip=\z@
  \rightskip=\@rightskip
  }
\def\normal@text{%
  \global\let\\=\@normalcr
  \global\leftskip=\z@
  \global\@rightskip=\z@
  \global\rightskip=\@rightskip
  \global\parfillskip=\@flushglue
  }
% \def\@writecount{%
  % \write\@mainaux{\string\global\string\n@authors=\the\c@author}%
  % }

\def\@pretitle@output#1{#1}
\def\@pretitle@hrule{\hrule}
\def\@pretitle@md{}
\def\@pretitle{???}
\long\def\pretitle@key@@#1#2{%
  \xdef\@pretitle{#1}%
  #2%
  }%
\long\def\pretitle@key@XML@#1#2{%
  \xdef\@pretitle{#1}%
  \ignorespaces
  }
\setpkgattr{pretitle}{sep}{. }
\newcommand\pretitle[2][]{%
  \xdef\pretitle@used{1}%
  \mdforvkt{main-article-info}{pretitle}{#2}%
  \@ifundefined{pretitle@key@#1@}
    {\vtxClassError{\fmt@name}{unknown pretitle option key '#1'}{}}
    {}%
  \@nameuse{pretitle@key@#1@}{#2}{%
    \mdforvkt{main-article-info}{title}{#2}%
    \gdef\@pretitle@md{#2\pretitle@sep}%
    \vspace*{\b@pretitle@skip}%
    \pretitle@size
    \@pretitle@output{#2}%
    \par
    \vskip6\p@
    \@pretitle@hrule
    \vskip12\p@
    }%
    \global\let\pretitle\undefined
  }
\def\title#1{%
  \bgroup
    \no@md@harm
    \xdef\@argi{#1}%
  \egroup
  \write@pdfinfo{\hy@title}{\@argi}%
  \mdforvkt{main-article-info}{title}{\@pretitle@md #1}%
  \beg@elem
  \title@note@fmt
  \add@tok\t@glob@notes{\title@note@fmt}%
  \proc@elem{title}{#1}%
  \def\title@notes{\the\t@loc@notes}%
  \title@fmt{\@title}{\title@notes}%
  \global\c@sv@footnote=\c@footnote
  \setcounter{footnote}{0}%
  \ignorespaces
  }

\newdimen\@@topskip \@@topskip=24\p@

\def\title@textsuperscript#1{\@textsuperscript{\normalfont#1}}
\def\titlenote@size{\fontsize{20}{20}\selectfont}

\ifnum\ios@journal@layout>\z@
  \def\title@textsuperscript#1{\hbox{$^{#1}$}}
  \let\titlenote@size\relax
\fi

\def\title@fmt#1#2{%
  \vspace*{\@@topskip}%
  \bgroup
    \title@size
    #1%
    \hbox{%
      \titlenote@size
      \title@textsuperscript{#2}%
      }%
    \par
  \egroup
  \vskip\a@title@skip
  }

\def\subtitle#1{%
  \beg@elem
  \proc@elem{subtitle}{#1}%
  \def\title@notes{\the\t@loc@notes}%
  \subtitle@fmt{\@subtitle}{\title@notes}%
  \ignorespaces
  }
\def\subtitle@fmt#1#2{%
  \bgroup
    \subtitle@size
    #1\,\hbox{$^{\mathrm{#2}}$}%
    \par
  \egroup
  \vskip\a@title@skip
  }
\def\title@note@fmt{\def\thefootnote{\arabic{footnote}}}

\def\oldreview#1#2#3{%
  \bgroup
    \footnotesize
    \textnormal
    \noindent
    \ifx.#1.%
    \else
      \textbf{Editor(s):} #1\\
    \fi
    \ifx.#2.%
    \else
      \textbf{Solicited review(s):} #2\\
    \fi
    \ifx.#3.%
    \else
      \textbf{Open review(s):} #3%
    \fi
    \par
    \vskip\a@title@skip
  \egroup
  }

\newcounter{review}
\newcounter{reviewauthor}
\@namedef{review@open@settings}{%
  \xdef\reviewertype{reviewer}%
  \xdef\reviewname{Open review(s)}%
  }
\@namedef{review@solicited@settings}{%
  \xdef\reviewertype{reviewer}%
  \xdef\reviewname{Solicited review(s)}%
  }
\@namedef{review@editor@settings}{%
  \xdef\reviewertype{editor}%
  \xdef\reviewname{Editor(s)}%
  }
\def\reviewname{???}
\def\reviewertype{???}
\def\reviewname@fmt#1{\noindent\textbf{#1:}}
\def\review@size{\normalfont\footnotesize}

\def\reviewer#1{%
  \bgroup
    \unskip\unkern%
    \ifnum\c@reviewauthor>\z@ ;\fi\
    \stepcounter{reviewauthor}%
    \let\prefix\@firstofone
    \let\fnms\@firstofone
    \let\inits\@firstofone
    \let\snm\@firstofone
    \let\suffix\@firstofone
    \let\ead@prefix@orcid\relax
    \let\ead@prefix@m@orcid\relax
    \def\orcid@output@fmt##1{(##1)}%
    \def\address##1{\unskip, ##1}%
    #1%
  \egroup
  \ignorespaces
  }%

\newenvironment{review}[1]%
  {%
    \ifnum\c@review>\z@
      \vskip-\a@title@skip
    \fi
    \stepcounter{review}%
    \setcounter{reviewauthor}{0}%
    \def\author{%
      \vtxClassError{\fmt@name}{^^J\string\reviewer{} command must be used inside \string{review\string} environment}{}%
      }%
    \review@size
    \@nameuse{review@#1@settings}%
    \reviewname@fmt{\reviewname}%
  }{%
    \par
    \vskip\a@title@skip
  }

\@namedef{ds@review@open@settings}{%
  \xdef\reviewertype{reviewer}%
  \ifnum\c@reviewauthor>\@ne
    \xdef\reviewname{Open reviews}%
  \else
    \xdef\reviewname{Open review}%
  \fi
  }
\@namedef{ds@review@solicited@settings}{%
  \xdef\reviewertype{reviewer}%
  \ifnum\c@reviewauthor>\@ne
    \xdef\reviewname{Solicited reviews}%
  \else
    \xdef\reviewname{Solicited review}%
  \fi
  }
\@namedef{ds@review@editor@settings}{%
  \xdef\reviewertype{editor}%
  \ifnum\c@reviewauthor>\@ne
    \xdef\reviewname{Editors}%
  \else
    \xdef\reviewname{Editor}%
  \fi
  }

\newtoks\dsreview@toks
\newdimen\dsreview@preskip \dsreview@preskip=15.5\p@
\newdimen\dsreview@postskip \dsreview@postskip=12\p@
\def\count@reviewers#1{%
  \setcounter{reviewauthor}{0}%
  \setbox\z@=\vbox{#1}%
  }
\long\def\dsreview#1#2\end#3{%
  \def\check@param@a{review}%
  \def\check@param@b{#3}%
  \ifx\check@param@a\check@param@b
    \def\@next@{%
      \global\dsreview@toks\expandafter{\the\dsreview@toks#2}%
      %% review author setup
      \def\author{%
        \vtxClassError{\fmt@name}{^^J\string\reviewer{} command must be used inside \string{review\string} environment}{}%
        }%
      \count@reviewers{\the\dsreview@toks}%
      \ifnum\c@review=\z@
        \vskip\dsreview@preskip
      \else
        \vskip-\dsreview@postskip
      \fi
      \stepcounter{review}%
      \@nameuse{ds@review@#1@settings}%
      \setcounter{reviewauthor}{0}%
      \review@size
      \csname begin@dsreview@hook\endcsname
      \reviewname@fmt{\reviewname}%
      \the\dsreview@toks
      \global\dsreview@toks{}%
      \par
      \vskip\dsreview@postskip
      \csname end@dsreview@hook\endcsname
      \end{#3}%
      }%
  \else
    \def\@next@{%
      \global\dsreview@toks\expandafter{\the\dsreview@toks#2\end{#3}}%
      \dsreview{#1}%
      }%
  \fi
  \@next@
  }

\let\review\dsreview
\let\endreview\enddsreview

\@ifundefined{short@journal}
  {\let\short@journal\@jid}
  {}

\csuse{\fmt@name @vkt@cmn@fm@add}

% AUG - author block
\newtoks\aug@toks
\def\add@to@aug@tok#1#2#3{%
  \global\aug@toks=\expandafter{\the\aug@toks #1#2{#3}}%
  }  
\def\collect@author#1#{%
  \global\advance\n@authors by 1
  \add@to@aug@tok{\author}{#1}% 
  }
\def\collect@address#1#{%
  \add@to@aug@tok{\address}{#1}% 
  }

\newenvironment{aug}{%
    \setbox\@tempboxa=\hbox\bgroup
    \let\author\collect@author
    \let\address\collect@address
  }{%
    \egroup
    \the\aug@toks
  }

\def\author{%
  \@ifnextchar[%
    {\author@optarg}
    {\author@optarg[]}%
  }

\def\author@optarg[#1]#2{%
  \global\ead@labels={}%
  \stepcounter{author}%
  \beg@elem
  \def\degs##1{##1}%
  \@ifundefined{prefix@\the\c@author @tmp}
    {\global\@namedef{prefix@\the\c@author @tmp}{}}
    {}%
  \def\prefix##1{%
    \protect\protected@write\@auxout{}{\string\global\string\@namedef{prefix@\the\c@author @tmp}{##1 }}%
    \if@make@lua@auth@\else
      ##1%
    \fi
    }%
  \def\fnms##1{%
    \protect\mdforvkt{main-article-info}{fnm-\the\c@author}{\@nameuse{prefix@\the\c@author @tmp}##1}%
    \cmd@for@aut@elem{fnms}{\the\c@author}{\@nameuse{prefix@\the\c@author @tmp}##1}%
    \cmd@for@aut@elem{fnmsmd}{\the\c@author}{##1}% for PDF metadata only, which don't have prefix
    \if@make@lua@auth@
      \if .##1.\else
        \vtx@directlua{%
          \short@journal_author[\the\c@author] = \short@journal_author[\the\c@author] or {} 
          \short@journal_author[\the\c@author].fnms = 'x'
          }%
      \fi
    \else
      ##1%
    \fi
    }%
  \def\inits##1{%
    \protect\mdforvkt{main-article-info}{inits-\the\c@author}{\@nameuse{prefix@\the\c@author @tmp}##1}%
    \cmd@for@aut@elem{inits}{\the\c@author}{\@nameuse{prefix@\the\c@author @tmp}##1}%
    \ignorespaces
    }%
  \def\snm##1{%
    \protect\protected@write\@auxout{}{\string\global\string\@namedef{snm@\the\c@author @tmp}{##1}}%
    \protect\mdforvkt{main-article-info}{snm-\the\c@author}{##1}%
    \cmd@for@aut@elem{snm}{\the\c@author}{##1}%
    \if@make@lua@auth@\else
      ##1%
    \fi
    }%
  \@ifundefined{snm@\the\c@author @tmp}
    {\@namedef{snm@\the\c@author @tmp}{}}
    {}%
  \def\suffix##1{%
    \protect\mdforvkt{main-article-info}{snm-\the\c@author}{\@nameuse{snm@\the\c@author @tmp}\space ##1}%
    \cmd@for@aut@elem{snm}{\the\c@author}{\@nameuse{snm@\the\c@author @tmp}\space ##1}%
    \if@make@lua@auth@\else
      ##1%
    \fi
    }%
  \def\roles##1{%
    \if@make@lua@auth@\else
      ##1%
    \fi
    }%
  \def\bio##1{%
    \csxdef{ios@bio@##1@label}{1}%
    }%
  \if@firstauthor
    \first@author
    \global\@firstauthorfalse
  \fi
  \@for\@tempa:=#1\do{%
    \expandafter\add@addressref\expandafter{\@tempa}%
    \if@make@lua@auth@
      \vtx@directlua{%
        \short@journal_author[\the\c@author] = \short@journal_author[\the\c@author] or {} 
        \short@journal_author[\the\c@author].addrefs = \short@journal_author[\the\c@author].addrefs or {}
        table.insert(\short@journal_author[\the\c@author].addrefs, '\@tempa')
        }%
    \fi
    }%
  \let\oldunderscore\_%
  \let\_\textunderscore
  \if@make@lua@auth@
    \vtx@directlua{%
      \short@journal_author[\the\c@author] = \short@journal_author[\the\c@author] or {} 
      }%
  \fi
  \proc@elem{author}{#2}%
  \let\_\oldunderscore
  \if@make@lua@auth@
    \setbox\@tempboxa=\hbox{\@author}%
    \vtx@directlua{%
      \short@journal_author[\the\c@author] = \short@journal_author[\the\c@author] or {} 
      % \short@journal_author[\the\c@author].locnotes = '\the\t@loc@notes'
      \short@journal_author[\the\c@author].eads = '\the\ead@labels'
      }%
    %
    \def\thanks##1{%
      \if .##1.\else
        \vtx@directlua{%
          \short@journal_author[\the\c@author] = \short@journal_author[\the\c@author] or {} 
          \short@journal_author[\the\c@author].thanks = 'x'
          }%
        \fi
      }%
    \setbox\@tempboxa=\hbox{#2}%
  \else
    \author@fmt{\the\c@author}{\the\t@loc@notes}{\@author}%
  \fi
  }%

\newtoks\ead@labels% csv list of all ead labels of the current author: e2,e3
\ead@labels={}
\ifdefined\HCode\else
  \def\append@ead@label#1=#2\relax{\expandafter\global\expandafter\ead@labels\expandafter =\expandafter{\the\ead@labels #2}}% \append@ead@label label=e1\relax
  \ifnum\ios@journal@layout=5\relax
    \AtBeginDocument{%
      \let\ead@sav\ead
      \renewcommand\ead[2][]{%
        \if.\the\ead@labels.%
        \else
          \expandafter\global\expandafter\ead@labels\expandafter =\expandafter{\the\ead@labels,}%
        \fi
        \append@ead@label #1\relax
        \ead@sav[#1]{#2}%
        }%
      }%
  \fi
\fi

\def\author@fmt#1#2#3{%
  \global\ead@labels={}%
  \@newelemtrue
  \ifnum\prev@elem=\e@author
    \global\@newelemfalse
  \fi
  \if@newelem
    \author@fmt@init
  \fi
  \edef\@tempb{#2}%
  \bgroup
    \auth@style
    \ifx\@tempb\@empty
      \hbox{#3}%
    \else
      \hbox{#3\auth@fn@skip\fm@textsuperscript{%
        \if@make@lua@auth@
          \color{Maroon}%
        \fi
        #2%
        }}%
    \fi
  \egroup
  \ifdefined\HCode\else
    \ifnum\ios@journal@layout=5\relax
      \hfill
      \def\ead@delim{}%
      \def\do##1{%
        \texttt{\ead@delim\csname ead@##1@value\endcsname}%
        \def\ead@delim{, }%
        }%
      \bgroup
        \small    
        \expandafter\docsvlist\expandafter{\the\ead@labels}%
      \egroup
      \break
    \fi
  \fi
  \output@author@sep
  }

\def\output@author@sep{%  
  \ifnum\n@authors>\z@
    \sep@by@num@in@seq
      {\the\c@author}%
      {\the\n@authors}%
      {\authors@septwo}%
      {\authors@sepmid}%
      {\authors@seplast}%
      {\authors@endgraf}%
  \fi
  }

\def\authors@septwo{ and }
\def\authors@sepmid{, }
\def\authors@seplast{ and }
\def\authors@endgraf{}
  
\dolayoutbyjid{com,sat}{
  \def\authors@septwo{}
  \def\authors@sepmid{}
  \def\authors@seplast{}
  \def\authors@endgraf{}
  }

%% x'so testavimui  
% \let\output@author@sep\empty
  
  
%%% SEP BY NUMBER IN SEQUENCE: FOR AUTHOR, ADDRESS, ETC SEP %%%
% #1 - current number in sequence (NUM)
% #2 - number of sequence elements (MAX)
% #3 - punctuation after element, when MAX = 2 and NUM < MAX        e.g., word 'and'
% #4 - punctuation after element, when MAX > 2 and (MAX-NUM) > 1    e.g., comma
% #5 - punctuation after element, when MAX > 2 and (MAX-NUM) = 1    e.g., comma and word 'and'
% #6 - punctuation after element, when MAX = NUM                    e.g., period
\def\sep@by@num@in@seq@error#1{\@latex@error{#1}\@ehc}
\def\sep@by@num@in@seq#1#2#3#4#5#6{%
    \ifnum#1>\numexpr#2\relax
        \sep@by@num@in@seq@error{First argument of command \string\@sep@by@num@in@seq\space 
            '#1' can not be greater than second '#2'}%
    \fi
    \ifnum#1<\@ne
        \sep@by@num@in@seq@error{First argument of command \string\@sep@by@num@in@seq\space 
            '#1' can not be smaller than 1}%
    \fi
    \ifnum#2>\z@
        \ifnum#2>\@ne
            \ifnum#2>\tw@
                \ifnum#1<\numexpr#2\relax
                    \ifnum\numexpr#2-#1\relax>\@ne
                        #4%
                    \else
                        #5%
                    \fi
                \else
                    #6%
                \fi
            \else
                \ifnum#1<\numexpr#2\relax
                    #3%
                \else
                    #6%
                \fi
            \fi
        \else
            #6%
        \fi
    \else
        \sep@by@num@in@seq@error{Number of elements in command \string\@sep@by@num@in@seq\space 
            '#2' must be greater than zero}%
    \fi
    }
  
\def\first@author{%
  \author@note@fmt
  \add@tok\t@glob@notes{\author@note@fmt}%
  }
  
\def\author@fmt@init{%
  \par
  \vskip\b@author@skip
  \authors@size
  \raggedright
  \leavevmode
  }

\if@make@lua@auth@
  \vtx@directlua{%
    % array length
    if (not table.getn) then
      table.getn = function(tab)
        ii = 0
        while not (tab[ii + 1] == nil) do
          ii = ii + 1
          end
        return ii;
        end
      end
    % single address output
    function make_addr(adr_ix, adr_val)
      locnotes = adr_val.locnotes or ''
      tex.print(bslash_chr .. 'makeatletter')
      tex.print(bslash_chr .. 'upd@elem{address}' .. perc_chr)
      tex.print(bslash_chr .. 'address@fmt{0}{' .. locnotes .. '}{' .. perc_chr)
        tex.print(bslash_chr .. 'csname \short@journal @raw@' .. adr_ix .. '@address' .. bslash_chr .. 'endcsname')
        tex.print('}' .. perc_chr)
      end
    %
    % single author output
    function make_aut(aut_ix, aut_val)
      fnms = aut_val.fnms or ''
      fnms_delim = bslash_chr .. ' '
      if (fnms == '') then
        fnms_delim = ''
        end
      eads = aut_val.eads or ''
      thanks = aut_val.thanks or ''
      locnotes = aut_val.locnotes or ''
      % thanks_cmd = ''
      if (not (thanks == '')) then
        % thanks_cmd = bslash_chr .. 'thanks{' .. thanks .. '}'
        % thanks_cmd = bslash_chr .. 'add@thanks *'
        % thanks_cmd = bslash_chr .. 'thefootnote'
        locnotes = '*'
        end
      %
      tex.print(bslash_chr .. 'makeatletter')
      % tex.print(bslash_chr .. 'def' .. bslash_chr .. 'thanksref' .. hash_chr .. '1' .. hash_chr .. hash_chr .. '{bslash_chr .. @gobble}' .. perc_chr)% \def\thanksref##1##{\@gobble}%
      tex.print(bslash_chr .. 'upd@elem{author}' .. perc_chr)
      tex.print(bslash_chr .. 'author@fmt{' .. tostring(aut_ix) .. '}{' .. locnotes .. '}{' .. perc_chr)
        % tex.print(fnms .. perc_chr)
        tex.print(bslash_chr .. 'csname \short@journal @fnms@' .. aut_ix .. '@author' .. bslash_chr .. 'endcsname')
        tex.print(fnms_delim .. perc_chr)
        tex.print(bslash_chr .. 'csname \short@journal @snm@' .. aut_ix .. '@author' .. bslash_chr .. 'endcsname')
        % tex.print(thanks_cmd .. perc_chr)
        tex.print(bslash_chr .. 'global' .. bslash_chr .. 'ead@labels={' .. eads .. '}' .. perc_chr)
        tex.print('}' .. perc_chr)
      end
    %
    % the whole author / address chunk
    function make_authors()
      % single address authors
      for adr_ix, adr_val in pairs(\short@journal_address) do
        addref = adr_val.addref or 'x'
        outfl = false
        for aut_ix, aut_val in pairs(\short@journal_author) do
          addrefs = aut_val.addrefs or {}
          if (table.getn(addrefs) == 1) and (addref == addrefs[1]) then
            outfl = true
            break
            end
          end
        if outfl then
          for aut_ix, aut_val in pairs(\short@journal_author) do
            addrefs = aut_val.addrefs or {}
            if (table.getn(addrefs) == 1) and (addref == addrefs[1]) then
              make_aut(aut_ix, aut_val)
              aut_val.done = true
              end
            end
          make_addr(adr_ix, adr_val)
          adr_val.done = true
          end
        end
      %  
      % multiple address authors
      for aut_ix, aut_val in pairs(\short@journal_author) do
        if not aut_val.done then
          make_aut(aut_ix, aut_val)
          aut_val.done = true
          addrefs = aut_val.addrefs or {}
          for _, addref in pairs(addrefs) do
            adr_ix = \short@journal_addr_map[addref]
            if adr_ix then
              adr_val = \short@journal_address[adr_ix]
              make_addr(adr_ix, adr_val)
              adr_val.done = true
              end
            end
          end
        end
      %
      % rest addresses
      for adr_ix, adr_val in pairs(\short@journal_address) do
        if not adr_val.done then
          make_addr(adr_ix, adr_val)
          end
        end
      end
    }%
\fi

\def\and{\unskip\ and }

\def\author@note@fmt{%
  \def\thefootnote{\sxarabic{footnote}}%
  }

\def\sxarabic#1{%
  \expandafter\ifcase\value{#1}%
  \or *%
  \or **%
  \or ***%
  \or ****%
  \or *****%
  \fi
  }
\def\syarabic#1{%
  \expandafter\ifcase\value{#1}%
  \or \zstar
  \or \zstar\zstar
  \or \zstar\zstar\zstar
  \or \zstar\zstar\zstar\zstar
  \or \zstar\zstar\zstar\zstar\zstar
  \fi
  }
\def\zstarshape{\fontencoding{U}\fontfamily{pzd}\fontseries{m}\fontshape{n}\selectfont}
\DeclareRobustCommand{\zstar}{{\fontencoding{U}\fontfamily{pzd}\fontseries{m}\fontshape{n}\selectfont I}}

\def\dept#1{#1}
\def\orgname#1{#1}
\let\institution\orgname
\def\cny#1{#1}
\let\email@href\@secondoftwo
\def\email{\@ifstar\@semail\@email}
\def\@semail#1{\email@href{#1}{#1}}
\def\email@prefix{e-mail:\ }
\def\@email#1{{\email@prefix\email@href{#1}{#1}}}

\def\email@href#1#2{\href{mailto:#1}{#1}}
\def\email@prefix{}

\csuse{\fmt@name @load@printead}
\appto\before@printead@hook{\def\ead@prefix@m@phone{Tels}}
\def\ead@hrefprefix@orcid{https://orcid.org/}
\def\orcid@value@prefix{https://orcid.org/}
\def\orcid@value@output#1{\orcid@value@prefix #1}

\def\ead@href#1#2{\href{#1}{#2}}%
\appto\@setead@hook{%
  \protected\def\textasciitilde{\noexpand\textasciitilde}%
  \protected\def\texttildelow{\noexpand\texttildelow}%
  \protected\def\textunderscore{\noexpand\textunderscore}%
  }

\def\address{%
  \@ifstar
    {\address@star}
    {%
      \@ifnextchar[%
        {\address@optarg}
        {\address@noptarg}%
    }%
  }
\if@make@lua@auth@
  \def\cmd@for@elem#1#2#3#4{% forms new command for an element of the author or address
    \global\@namedef{\short@journal @#1@#2@#3}{#4}%
    }
  \def\address@optarg[#1]#2{%
    \refstepcounter{address}%
    \beg@elem
    \cmd@for@elem{raw}{\the\c@address}{address}{#2}%
    \proc@elem{address}{#2}%
    \setbox\@tempboxa=\vbox{\@address}%
    \vtx@directlua{%
      \short@journal_address[\the\c@address] = \short@journal_address[\the\c@address] or {} 
      \short@journal_address[\the\c@address].addref = '#1'
      % \short@journal_address[\the\c@address].locnotes = '\the\t@loc@notes'
      \short@journal_addr_map['#1'] = \the\c@address
      }%
    \ignorespaces
    }%
  \def\address@noptarg#1{%
    \address@optarg[\the\c@address]{#1}%
    }%
  \def\address@star#1{%
    \address@optarg[\the\c@address]{#1}%
    }%
\else
  \def\address@optarg[#1]#2{%
    \ifnum\ios@journal@layout=5\else
      \refstepcounter{address}%
    \fi
    \beg@elem
    \proc@elem{address}{#2}%
    \address@fmt{\the\c@address}{\the\t@loc@notes}{\@address}\label{#1}%
    \ignorespaces
    }
  \def\address@noptarg#1{%
    \refstepcounter{address}%
    \beg@elem
    \proc@elem{address}{#1}%
    \address@fmt{\z@}{\the\t@loc@notes}{\@address}%
    \ignorespaces
    }
  \def\address@star#1{%
    \beg@elem
    \proc@elem{address}{#1}%
    \address@fmt{\m@ne}{\the\t@loc@notes}{\@address}%
    \ignorespaces
    }
\fi
\def\theaddress{\alph{address}}
\def\address@fmt#1#2#3{%
  \@newelemtrue
  \ifnum\prev@elem=\e@address
    \@newelemfalse
  \fi
  \if@newelem
    \address@fmt@init
  \fi
  \bgroup
    \parskip=\z@
    \noindent
    \raggedright
    \address@size
    \ifnum#1=\z@
      #3\,\fm@textsuperscript{#2}\space
    \else
      \ifnum#1=\m@ne
        \phantom{\fm@textsuperscript{\theaddress}\,}#3\,\fm@textsuperscript{#2}%
      \else
        \fm@textsuperscript{\theaddress}\,#3\,\fm@textsuperscript{#2}%
      \fi
    \fi
    \par
  \egroup
  }
\def\address@fmt@init{%
  \def\@currentlabel{\theaddress}%
  \ifnum\ios@journal@layout=5\else
    \par
    \vskip 2\p@ plus 1\p@ minus 1\p@
  \fi
  }

\newskip\abstract@skip \abstract@skip=48pt
\def\abstract{%
  \@ifnextchar[%
    {\@abstract}
    {\@abstract[]}%
  }
\def\@abstract[#1]{%
  \let\c@footnote\c@sv@footnote
  \gdef\thefootnote{\arabic{footnote}}%
  \def\parttitle##1{%
    \endgraf
    \noindent
    {\bfseries\MakeUppercase{##1}:}%
    }%
  \global\@hasabstracttrue
  \if@nameyear
  \else
    \@if@vtex@in@house{%
      \def\abstract@cite@prompt{\string\abstract}%
      \let\cite\abstract@cite
      }{}%
  \fi
  \hyphenpenalty=\sv@hyphenpenalty
  \global\setbox\t@abstract=\vbox\bgroup
    \hsize=\abstract@width
    \ifnum\ios@journal@layout=5\relax
      \bgroup
        \center\large\abstractname@fmt{\abstractname}\par
      \egroup
      \par
      \vskip\abstract@name@skip
    \fi
    \justify@on
    \abstract@size
    \abstract@indent
    \ifdefined\HCode\else
      \everypar{\abstract@indent}%
    \fi
    \ifnum\ios@journal@layout=5\else
      \abstractname@fmt{\abstractname}%
    \fi
    \ignorespaces
  }
\def\endabstract{%
    \par
  \egroup
  }
\def\abstract@cite{%
  \@ifnextchar[%
    {\@abstract@cite}
    {\@abstract@cite[]}%
  }
\def\abstract@cite@prompt{\string\abstract}
\def\@abstract@cite[#1]#2{%
  \def\byear##1{%
    \gdef\@abstract@cite@byear{}%
    \vtex@citex@start{}{}{}%
    \hyper@natlinkstart{#2}##1\hyper@natlinkend
    \vtex@citex@end
    }% 
  \let\@abstract@cite@byear\undefined
  \if@filesw
    \immediate\write\@auxout{\string\citation{#2}}%
  \fi
  \setbox\@tempboxa=\hbox{#1}%
  \@ifundefined{@abstract@cite@byear}
    {\@latex@error{\abstract@cite@prompt\space\string\cite: use the following syntax \string\cite[\string\byear{xxxx}]{label}}{}}
    {}%
  #1%
  }

\newskip\keywords@skip \keywords@skip=12pt
\newskip\keywords@noabs@skip \keywords@noabs@skip=22pt
\newcounter{kwd}
\def\@kwd@output#1{#1}
\def\kwd#1{%
  \stepcounter{kwd}%
  \ifnum\c@kwd=\@ne
    \@kwd@output{#1}%
  \else
    \unskip, %
    \@kwd@output{#1}%
    \addto@keywords@list{, }%
  \fi
  \addto@keywords@list{#1}%
  \mdforvkt{main-article-info}{keyword-\thekwd}{#1}%
  }
\def\sep{\unskip, }
\global\@haskeywordsfalse
\newdimen\dp@t@keywords
\def\keyword{%
  \global\@haskeywordstrue
  \global\setbox\t@keywords=\vbox\bgroup
    \ifnum\jobstart@date@info>20211001\relax
      \if@hasabstract
        %\vskip\keywords@skip
        \vrule\@width\z@\@height\dimexpr\keywords@skip+0.5\baselineskip\relax\@depth\z@
      \else
        %\vskip\keywords@noabs@skip
        \vrule\@width\z@\@height\dimexpr\keywords@noabs@skip+0.5\baselineskip\relax\@depth\z@
      \fi
    \fi
    \hsize=\abstract@width
    \justify@on
    \keywords@size
    \keywords@indent
    \ifnum\jobstart@date@info>20211001\relax
      \hskip\parindent
    \fi
    \keywordsname@fmt{\keywordsname}%
    \keywordsname@sep
    \ignorespaces
  }
\def\endkeyword{%
    \par
  \egroup
  \global\dp@t@keywords=\dp\t@keywords
  }
\def\keywords#1{%
  \begin{keyword}%
  #1%
  \end{keyword}%
  }

\def\runningtitle#1{\vtxClassError{\fmt@name}{The command \string\runningtitle\space is obsolete, use \string\runtitle\space instead}{}}
\def\runtitle#1{\gdef\@runtitle{#1}}
\def\@runtitle{}
\def\runningauthor#1{\vtxClassError{\fmt@name}{The command \string\runningauthor\space is obsolete, use \string\runauthor\space instead}{}}
\def\runauthor#1{%
  \bgroup
    \def\etal{et al.}%
    \gdef\@runauthor{#1\@runsep}%
  \egroup
  }
\def\@runauthor{}
\def\runningsep#1{\gdef\@runsep{#1}}
\def\@runsep{\ /\ }
\def\journal#1{\gdef\@journal{#1}}
\@ifundefined{@journal}
  {\gdef\@journal{Journal not defined}}
  {}
\def\volume#1{%
  \mdforvkt{main-article-info}{volume}{#1}%
  \gdef\@volume{#1}%
  }
\def\@volume{0}
\def\issue#1{%
  \ios@check@supp@s#1\@nil
  \mdforvkt{main-article-info}{issue}{#1}%
  \gdef\@issue{#1}%
  }
\def\@issue{0}
\let\special@supplement@letter\@empty
\def\ios@check@supp@s#1#2\@nil{%
  \ifstrequal{#1}{s}
    {%
      \xdef\special@supplement@letter{S}%
      \protected@write\@auxout{}{\string\gdef\string\special@supplement@letter{S}}%
      \global\let\ios@old@thepage\thepage
      \gdef\thepage{\special@supplement@letter\ios@old@thepage}%
    }{}%
  }

\newcount\@@pubyear
\def\@pubyear{\the\@@pubyear}
\newcount\@@copyear
\@@copyear=\@@pubyear 
\def\@copyear{\the\@@copyear}
\ifnum\@copyear<2000\relax
  \advance\@@copyear-1900
\else
  \advance\@@copyear-2000
\fi

\def\pubyearto@error{%
  \vtxClassError{\fmt@name}{\string\pubyearto{...}\space is forbidden, use 
    \string\pubyearfrom{...}\space instead}{}%
  }
\AtBeginDocument{%
  \ifcsundef{@pubyearto}
    {}
    {%
      \ifnum\@pubyearto>\z@
        \pubyearto@error
      \fi
    }%
  }% for covers
\def\pubyearto#1{\pubyearto@error}% for normal articles
\def\@pubyearfrom@fmt{}
\def\pubyearfrom#1{%
  \gdef\@pubyearfrom{#1}%
  \gdef\@pubyearfrom@fmt{\@pubyearfrom/}%
  \ifnum#1<2000\relax
    \xdef\@copyearfrom{\the\numexpr#1-1900\relax}%
  \else
    \xdef\@copyearfrom{\the\numexpr#1-2000\relax}%
  \fi
  }% / could be easily hooked
\@onlypreamble\pubyearfrom
\def\pubyear#1{%
  \global\@@pubyear=#1\relax
  \global\@@copyear=\@@pubyear 
  \mdforvkt{main-article-info}{pubyear}{#1}%
  \ifnum\@copyear<2000\relax
    \global\advance\@@copyear-1900
  \else
    \global\advance\@@copyear-2000
  \fi
  \ignorespaces
  }
\pubyear{\number\year}
\def\the@copyear{%
  \ifnum\@copyear<10\relax
    0%
  \fi
  \@copyear
  }
\def\the@copyearfrom{%
  \ifcsundef{@copyearfrom}
    {}
    {%
      \ifnum\@copyearfrom<10\relax
        0%
      \fi
      \@copyearfrom/%
    }%
  }

\def\firstpage#1{%
  \mdforvkt{main-article-info}{firstpage}{#1}%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@firstpage{#1}%
    \gdef\@lastpage{#1}%
    \global\c@page=#1%
    \ignorespaces
  \fi
  }
\def\@firstpage{1}
\def\lastpage#1{%
  \mdforvkt{main-article-info}{lastpage}{#1}%
  \def\@tempa{#1}%
  \ifx\@tempa\@empty
  \else
    \gdef\@lastpage{#1}%
    \ignorespaces
  \fi
  }
\def\@lastpage{0}
\def\@pagerange{1--0}

\let\old@pagenumbering\pagenumbering
\def\pagenumbering#1{%
  \gdef\@page@numbering{#1}%
  \old@pagenumbering{#1}%
  }
\def\thefirstpage{%
  \csname @\@page@numbering\endcsname
  \@firstpage
  }
\def\thelastpage{%
  \csname @\@page@numbering\endcsname
  \@lastpage
  }

\long\def\convertas#1#2{#2}
\def\aid#1{\gdef\@aid{#1}}
\def\SSDI#1{\gdef\@ssdi{#1}}
\def\@ssdi{000000-00}
\def\issn#1{\gdef\@issn{#1}}
\def\price#1{\gdef\@price{#1}}
\def\date#1{\gdef\@date{#1}}
\def\@date{\today}
\def\empty@data{\@nil}
\def\doi#1{\gdef\@doi{#1}}
\def\doi@base{https://doi.org/}
\let\@doi\relax

\csuse{\fmt@name @articletype}
%% <<< Frontmatter <<<

\newcommand\backmatter{\goodbreak}

\csdef{known@biography@jhs@project}{1}
\csdef{known@biography@web@project}{1}
\define@key{biogr}{title}{\def\biographygroup@title{#1}}
\newenvironment{biographygroup}[1][title=Biography]%
  {%
    \ifcsundef{known@biography@\@jid @project}
      {%
        \vtxClassError{\fmt@name}{the biographygroup environment hasn't been defined for '\@jid' project}{}%
      }{}%
    \setkeys{biogr}{#1}%
    \section*{\biographygroup@title}%
    \global\let\ios@bio@save@label\label
  }{}
\def\biography@size{}
\def\biography@figsep{1em}
\def\biography@figlift{.7em}
\def\biography@skip{1em}
\def\biography@setpar{%
  \unkern\unpenalty\unskip
  \hfill\break\mbox{\kern\parindent}%
  }%
\newbox\bio@tempboxa
\newcounter{biography}
\newenvironment{biography}
  {%
    \ifcsundef{known@biography@\@jid @project}
      {%
        \vtxClassError{\fmt@name}{the biography environment hasn't been defined for '\@jid' project}{}%
      }{}%
    \addvspace{\biography@skip}%
    \stepcounter{biography}%
    \biography@size
    %% dummy default figure height and num of line during 1st compilation
    \ifcsundef{bio@photo@\the\c@biography @ht}
      {% 
        \csdef{bio@photo@\the\c@biography @ht}{5pc}%
        \def\@tmp@bio@num@of@lines{1}%
      }{}%
    \def\photofilename##1{%
      %% box for figure dimensions
      \setbox\bio@tempboxa\hbox{\includegraphics{##1}}%
      %% figure height to aux (we need it when biography text is very short)
      %% figure usage hook too
      \protected@write\@auxout{}{%
          \string\csgdef{bio@photo@\the\c@biography @ht}{\the\ht\bio@tempboxa}%
          \string\csgdef{bio@photo@\the\c@biography @used}{1}%
          }%
      %% calculating figure height with space around (striping space from baseline [bs-.7em],
      %% i.e. space from baseline to real text)
      \@tempdima=\dimexpr\ht\bio@tempboxa+\biography@figsep-(\baselineskip-.7em)\relax
      %% calculating, how many lines we have indent due figure
      \@tempdimb\z@ \@tempcnta\z@
      \@whiledim\@tempdimb<\@tempdima\do{%
        \advance\@tempcnta\@ne
        \advance\@tempdimb\baselineskip
        }%
      %% putting figure
      \noindent\llap{%
        \raisebox{\the\dimexpr-\ht\bio@tempboxa+\biography@figlift\relax}%
          [\z@][\z@]{\copy\bio@tempboxa}%
        \hskip\biography@figsep
        }%
      %% making parshape settings
      \@tempcntb=\z@
      \@tempdima=\dimexpr\wd\bio@tempboxa+\biography@figsep\relax
      \edef\bio@tmp{\the\numexpr\the\@tempcnta+1\relax\space}%
      \@whilenum\@tempcntb<\@tempcnta\do{%
        \edef\bio@tmp{\bio@tmp\the\@tempdima\space\the\dimexpr\hsize-\@tempdima\relax\space}%
        \advance\@tempcntb\@ne
        }%
      %% saving number of indenting lines
      \xdef\@tmp@bio@num@of@lines{\the\@tempcntb}%
      %% doing parshape
      \edef\bio@tmp{\bio@tmp 0pt \the\hsize}%
      \expandafter\parshape\bio@tmp%
      \ignorespaces
      }%
    \def\label##1{%
      \ifcsdef{ios@bio@label@\the\c@biography @used}
        {%
          \vtxClassError{\fmt@name}{only one \string\label{...} is allowed in biography, 
            \string\label{##1} is excess}{}%
        }{%
          \csxdef{ios@bio@label@\the\c@biography @used}{1}%
          \ifcsundef{ios@bio@##1@label}
            {%
              \vtxClassError{\fmt@name}{command \string\bio{...}\space not used for label: ##1}{}%
            }{}%
        }%
      \ios@bio@save@label{##1}%
      \ignorespaces
      }%
    %% adding all biography to box for situations when text is very short
    \setbox\@tempboxa\vbox\bgroup
      % if figure is not used, paragraph alternation is not necessary
      \ifcsundef{bio@photo@\the\c@biography @used}%
        {\noindent}%
        {\@setpar\biography@setpar}%
      \ignorespaces
  }{%
      \ifhmode
        \endgraf
      \fi
    \egroup
    \ifcsdef{ios@bio@label@\the\c@biography @used}
      {}
      {%
        \vtxClassError{\fmt@name}{\string\label{...}\space must be used in biography}{}%
      }%
    %% if photofilename is not used, just unvbox vbox content
    \ifcsundef{bio@photo@\the\c@biography @used}%
      {%
        \unvbox\@tempboxa
      }{%
        %% if biography text is very short, we're making box with figure height
        \ifdim\dimexpr\ht\@tempboxa+\dp\@tempboxa\relax<\csuse{bio@photo@\the\c@biography @ht}%
        \vbox to \csuse{bio@photo@\the\c@biography @ht}{\unvbox\@tempboxa\vfill}%
        %% if biography text is long, we're splitting box (by number of indenting lines)
        %% and putting first part as (unbreakable) box, second as unvbox
        \else
          \setbox\bio@tempboxa=\vsplit\@tempboxa to \dimexpr\baselineskip*\@tmp@bio@num@of@lines\relax
          \box\bio@tempboxa
          \ifvoid\@tempboxa
          \else
            \unvbox\@tempboxa
          \fi
       \fi
    }%
  }

\gdef\refname{References}
\newcommand\figurename{Fig.}
\newcommand\tablename{Table}
\newcommand\algorithmname{Algorithm}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\newcommand\abstractname@fmt[1]{{\bfseries #1\abstractname@sep}}
\newcommand\keywordsname{Keywords}
\def\copyright@sign{\copyright}

\def\pretitle@size{\LARGE}
\def\title@size{\Huge}
\def\subtitle@size{\large\itshape}
\def\authors@size{\large}
\def\abstract@size{\small}
\def\address@size{\normalsize\itshape}

\newcommand\today{}
\edef\today{%
  \ifcase\month
  \or January%
  \or February%
  \or March%
  \or April%
  \or May%
  \or June%
  \or July%
  \or August%
  \or September%
  \or October%
  \or November%
  \or December%
  \fi
  \space\number\day, \number\year
  }

\@twosidetrue
\pagenumbering{arabic}
\frenchspacing
\init@settings
\if@twocolumn
  \if@load@flushend
    \IfFileExists{flushend_v35.sty}
      {\RequirePackage[nospread]{flushend_v35}}
      {\RequirePackage[nospread]{flushend}}
    %% Ijungiamas puslapio lauzimo metu ismetamu noad'u issaugojimas.
    %% Juos galima panaudoti flushend ismestam tarpu sugrazinti per \atColsBreak{\pagediscards}
    \savingvdiscards=1     
  \fi
\fi

\if@twocolumn
  \setlength\tablewidth{\columnwidth}
\else
  \setlength\tablewidth{\textwidth}
  \ifnum\ios@journal@layout>\z@
    \def\footnoterule@width{144.5\p@}
  \fi
\fi
\pagestyle{headings}

%% esm environment for xml
\def\esm@title{Supplementary data}
\def\esm@titlesep{}
\def\esm@titlefmt{%
  \section*{\esm@title}%
  \esm@titlesep
  }
%\def\extract@skip#1\p@#2\relax{#1}
%\def\section@beforeskip@val{\expandafter\extract@skip\section@beforeskip\relax}
%\def\section@afterskip@val{\expandafter\extract@skip\section@afterskip\relax}
%\def\esm@skip@suppress{\dimexpr-\section@beforeskip@val\p@-\section@afterskip@val\p@\relax}
\def\esm@skip@suppress{\dimexpr-3\baselineskip\relax}
\define@key{esm}{title}{%
  \def\esm@title{%
    #1%
    \if.#1.%
      \vskip\esm@skip@suppress
    \fi
    }%
  }
\def\esm{%
  \@ifnextchar[%
    \@esm
    {\@esm[]}%
  }
\def\@esm[#1]{%
  \bgroup
  \if.#1.%
  \else
    \setkeys{esm}{#1}%
  \fi
  \par
  \def\esmtitle{\@gobble}%
  \def\esmdescription{\@firstofone}%
  \def\esmfiletype{\@gobble}%
  \def\esmfilename{\@gobble}%
  \esm@titlefmt
  }
\def\endesm{\egroup}

\def\notes@skip{%
  \section@beforeskip
  \vskip\b@section@skip
  }
\def\notes@size{\normalsize}
\let\notes@end@hook\@empty
\newbox\notes@box
\newenvironment{notes}
  {%
    \xdef\has@notes{1}%
    \global\setbox\notes@box\vbox\bgroup
      \notes@size
  }{%
    \egroup
  }

\ifdefined\HCode
  % hyperref definitions required for the package cleveref
  \let\H@refstepcounter\refstepcounter
  \let\cref@old@H@refstepcounter\H@refstepcounter
\else
  \RequirePackage{nameref}
\fi
\RequirePackage{hyperref}
\csuse{\fmt@name @load@hypersetup}
\csuse{\fmt@name @load@vtexhdvips}
\setcounter{tocdepth}{3}
% \RequirePackage{breakurl}
\csuse{\fmt@name @load@breakurl@hook}
\csuse{\fmt@name @load@pagerefsout}

\let\ios@H@@footnotetext\H@@footnotetext
\long\def\H@@footnotetext#1{%
  \bgroup
    \def\@pdfview{XYZ}%
    \def\@pdfviewparams{ pdf@hoff pdf@voff \strip@pt\baselineskip\space 72.27 div 72 mul add null}%
    \ios@H@@footnotetext{#1}%
  \egroup
  }

\let\texttildelow\textasciitilde
\preto\hyper@normalise{\let\texttildelow\hyper@tilde}

\@namedef{ios@section@level@1@sep}{}
\@namedef{ios@section@level@2@sep}{}
\@namedef{ios@section@level@3@sep}{}
\@namedef{ios@section@level@4@sep}{}
\@namedef{ios@section@level@5@sep}{}
\ifnum\ios@journal@layout>\z@
  \@namedef{ios@section@level@4@sep}{.}
  \@namedef{ios@section@level@5@sep}{.}
\fi
\def\ios@section@numlevel{0}
\def\ios@section@namelevel{undef}
\newcounter{ios@section@star}
\AtBeginDocument{%
  \global\let\ios@Sectionformat\Sectionformat
  \global\let\ios@startsection\@startsection
  \def\@startsection#1#2#3#4#5#6{%
    \xdef\ios@section@namelevel{#1}%
    \xdef\ios@section@numlevel{#2}%
    \ios@startsection{#1}{#2}{#3}{#4}{#5}{#6}%
    }%
  \gdef\Sectionformat#1#2{%
    \ios@Sectionformat{#1\@nameuse{ios@section@level@\ios@section@numlevel @sep}}{#2}%
    }%
  \let\ios@NR@ssect\NR@ssect
  \def\NR@ssect#1#2#3#4#5{%
    \stepcounter{ios@section@star}%
    \def\ios@section@star@id{\ios@section@namelevel *.\the\c@ios@section@star}%
    \pdfbookmark[\ios@section@numlevel]{#5}\ios@section@star@id
    \ios@NR@ssect{#1}{#2}{#3}{#4}{#5}%
    }%
  }

\csuse{\fmt@name @hypermark@settings}
\def\write@pdfinfo#1#2{}
\newtoks\authors@list
\def\addto@authors@list#1{%
  \begingroup%
    \no@md@harm%
    \xdef\@act{\global\noexpand\authors@list{\the\authors@list#1}}%
    \@act
  \endgroup
  }
\newtoks\keywords@list
\def\addto@keywords@list#1{%
  \begingroup
    \no@md@harm
    \xdef\@act{\global\noexpand\keywords@list{\the\keywords@list#1}}%
    \@act
  \endgroup
  }

\csuse{\fmt@name @vkt@cmn@settings}
\csuse{\fmt@name @load@htmisc}

\DeclareRobustCommand*\textsubscript[1]{%
  \@textsubscript{\selectfont#1}%
  }
\def\@textsubscript#1{%
  {\m@th\ensuremath{_{\mbox{\fontsize\sf@size\z@#1}}}}%
  }
\let\tsup\textsuperscript
\let\tsub\textsubscript

\common@layout@settings
\@nameuse{init@ios@journal@\ios@journal@layout @layout}
\@nameuse{init@ios@journal@\@JID @settings}

\AtBeginDocument{\mdforvkt{main-article-info}{fullxml}{1}}%
\AtBeginDocument{\gdef\multi@issue@sep@diff@one{,}}%

\csuse{\fmt@name @urilist}
\csuse{\fmt@name @speciss}

%% >>> startlocaldefs, endlocaldefs >>>
\let\check@localdefs@status\relax
\def\startlocaldefs{%
  \gdef\check@localdefs@status{%
    \vtxClassError{\fmt@name}{Missing command \string\endlocaldefs!}{}%
    }%
  \gdef\startlocaldefs{%
    \vtxClassError{\fmt@name}{Command \string\startlocaldefs already exists!}{}%
    }%
  \makeatletter
  }
\def\endlocaldefs{%
  \makeatother
  \ifx\check@localdefs@status\relax
    \gdef\check@localdefs@status{%
      \vtxClassError{\fmt@name}{Missing command \string\startlocaldefs!}{}%
      }%
  \else
    \global\let\check@localdefs@status\relax
  \fi
  \gdef\endlocaldefs{%
    \vtxClassError{\fmt@name}{Command \string\endlocaldefs already exists!}%
    }%
  }
\AtBeginDocument{\check@localdefs@status}%
%% <<< startlocaldefs, endlocaldefs <<<

%% commands for tex4ht
\let\xmlEntryAlign\@gobble
\let\xmlstring\@gobble
\let\noxml\@firstofone
\def\floatanchor#1{}
\let\Mathbin\mathbin
\let\Mathrel\mathrel
\let\Mathord\mathord
\let\Mathop\mathop
\let\Mathopen\mathopen
\let\Mathclose\mathclose
\let\Mathpunct\mathpunct

%% definitions execution after \begin{document} (for tex4ht)
\ifdefined\HCode
  \bgroup
    \catcode`#=12\relax  
    \catcode`&=6\relax 
    \gdef\startlocaldefs{%
      \catcode`#=12\relax
      \makeatletter
      \@startlocaldefs
      }
    \gdef\@startlocaldefs@toks{}%
    \long\gdef\@startlocaldefs&1\endlocaldefs{%
      \long\xdef\@startlocaldefs@toks{\detokenize{&1}}%
      \makeatother
      \catcode`#=6\relax
      }  
  \egroup
  \appto\document{%
    \makeatletter
    \expandafter\scantokens\expandafter{\@startlocaldefs@toks}%
    \makeatother
    }%
\fi

\csuse{\fmt@name @trimbox@settings}

\let\eqref@fontshape\@firstofone
\def\eqref#1{\eqref@fontshape{(\ref{#1})}}%

\csuse{\fmt@name @check@floats}

%% \bid{doi={...}} syntax
\let\doi@arg\@empty
\let\pmid@arg\@empty
\define@key{bid}{doi}{%
  \def\doi@arg{#1}%
  \edef\bid@exists{1}%
  }
\define@key{bid}{pmid}{%
  \def\pmid@arg{#1}%
  \edef\bid@exists{1}%
  }
\def\bbl@bid#1{%
  \bgroup
    \setkeys{bid}{#1}%
    \@ifundefined{bid@exists}
      {}
      {\unskip}%
    \csname startchkuriskip\endcsname
    \csname startchecklinksskip\endcsname
    \ifx\pmid@arg\@empty
    \else
      \space
      \split@urlmacro{pmid}{\pmid@arg}%
      .%
    \fi
    \ifx\doi@arg\@empty
    \else
      \space
      \split@urlmacro{doi}{\doi@arg}%
      .%
    \fi
    \csname endchecklinksskip\endcsname
    \csname endchkuriskip\endcsname
  \egroup
  }
\setpkgattr{biddoi}{prefix}{doi}
\setpkgattr{biddoi}{sep}{:}
\setpkgattr{bidpmid}{prefix}{PMID}
\setpkgattr{bidpmid}{sep}{:}
\setpkgattr{pmid}{base}{https://www.ncbi.nlm.nih.gov/pubmed/}
\csuse{\fmt@name @load@spliturlmacro}
\let\bid\bbl@bid
\csuse{\fmt@name @bptdefs}
\let\bid@doi@space\relax
\def\bid@doi@sep{:}

\def\level@we@seek{subsubsection}
\def\space@after@section{%
  \ifvmode
    \if@nobreak
      \vskip-\abovedisplayskip
      \vskip-\baselineskip
      \ifx\ios@section@namelevel\level@we@seek
        \vskip13.0pt plus 0.5pt minus 0.1pt
      \fi      
    \fi
  \fi
  }
\pretocmd{\equation}{\space@after@section}{}{}
\expandafter\expandafter\expandafter\pretocmd\expandafter\csname equation*\endcsname{\space@after@section}{}{}
\pretocmd{\eqnarray}{\space@after@section}{}{}

%% left aligned gathered (from amsmath.sty)
\newcommand\ios@gathered[1][c]{%
  \RIfM@\else
    \nonmatherr@{\begin{gathered}}%
  \fi
  \null\,%
  \if #1t\relax
    \vtop
  \else
    \if#1b\relax
      \vbox
    \else
      \vcenter
    \fi
  \fi
  \bgroup
    \Let@\chardef\dspbrk@context\@ne
    \restore@math@cr
    \spread@equation
    \ialign
    \bgroup
      \strut@$\m@th\displaystyle##$\hfil
      \crcr
  }
\let\gathered\ios@gathered

%% >>> book-review >>>
\newenvironment{product}
  {}{}
\def\bpages#1{#1}
\def\bprice#1{#1}
%% <<< book-review <<<

\endinput
%%
%% End of file `iosart2x.cls'.
