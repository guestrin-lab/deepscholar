%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% cohere.cls
%
% LaTeX class template for cohere.for.ai
%
% code by E. A. Farina latex.guru
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\def\PATHCLASS{LaTeXclass}
\ProvidesClass{\PATHCLASS/cohere}[2025/06/11 v2.2 LaTeX class template for cohere.for.ai. publications]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prior packages
    \RequirePackage{keyval}
    \RequirePackage{xcolor}
    \RequirePackage{etoolbox}
    \RequirePackage{calc}
    \RequirePackage{changepage}
    \RequirePackage{tikz}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \LoadClass{article}
    \newbool{maut}\boolfalse{maut}
    \DeclareOption{multiauthors}{\booltrue{maut}}
    \newcommand\@ptsize{}
    \newif\if@restonecol
    \if@compatibility\else
      \DeclareOption{a4paper}
         {\setlength\paperheight {297mm}%
          \setlength\paperwidth  {210mm}}
      \DeclareOption{letterpaper}
         {\setlength\paperheight {11in}%
          \setlength\paperwidth  {8.5in}}
      \DeclareOption{landscape}
         {\setlength\@tempdima   {\paperheight}%
          \setlength\paperheight {\paperwidth}%
          \setlength\paperwidth  {\@tempdima}}
    \fi
    \if@compatibility
      \renewcommand\@ptsize{0}
    \else
      \DeclareOption{10pt}{\renewcommand\@ptsize{0}}
    \fi
    \DeclareOption{11pt}{\renewcommand\@ptsize{1}}
    \DeclareOption{12pt}{\renewcommand\@ptsize{2}}
    \if@compatibility\else
      \DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
    \fi
    \DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
    % draft
    \newbool{cdraft}\boolfalse{cdraft}
    \DeclareOption{cdraft}{\booltrue{cdraft}}
    \DeclareOption{cfinal}{\boolfalse{cdraft}}
    \DeclareOption{draft}{\setlength\overfullrule{5pt}}
    \DeclareOption{final}{\setlength\overfullrule{0pt}}
    %
    \newbool{twocol}
    \if@compatibility\else
      \DeclareOption{onecolumn}{\boolfalse{twocol}\@twocolumnfalse}
    \fi
    \DeclareOption{twocolumn}{\booltrue{twocol}\@twocolumntrue}
    \DeclareOption{leqno}{\input{leqno.clo}}
    \DeclareOption{fleqn}{\input{fleqn.clo}}
    \DeclareOption{openbib}{%
      \AtEndOfPackage{%
        \renewcommand\@openbib@code{%
          \advance\leftmargin\bibindent
          \itemindent -\bibindent
          \listparindent \itemindent
          \parsep \z@
          }%
        \renewcommand\newblock{\par}}%
    }
    \ExecuteOptions{letterpaper,11pt,oneside,twocolumn,final}
    \ProcessOptions
    \input{size1\@ptsize.clo}
    \setlength\lineskip{1\p@}
    \setlength\normallineskip{1\p@}
    \renewcommand\baselinestretch{}
    \setlength\parskip{0\p@ \@plus \p@}
    \@lowpenalty   51
    \@medpenalty  151
    \@highpenalty 301
    \setcounter{topnumber}{2}
    \renewcommand\topfraction{.7}
    \setcounter{bottomnumber}{1}
    \renewcommand\bottomfraction{.3}
    \setcounter{totalnumber}{3}
    \renewcommand\textfraction{.2}
    \renewcommand\floatpagefraction{.5}
    \setcounter{dbltopnumber}{2}
    \renewcommand\dbltopfraction{.7}
    \renewcommand\dblfloatpagefraction{.5}
    \if@twoside
      \def\ps@headings{%
          \let\@oddfoot\@empty\let\@evenfoot\@empty
          \def\@evenhead{\thepage\hfil\slshape\leftmark}%
          \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
          \let\@mkboth\markboth
        \def\sectionmark##1{%
          \markboth {\MakeUppercase{%
            \ifnum \c@secnumdepth >\z@
              \thesection\quad
            \fi
            ##1}}{}}%
        \def\subsectionmark##1{%
          \markright {%
            \ifnum \c@secnumdepth >\@ne
              \thesubsection\quad
            \fi
            ##1}}}
    \else
      \def\ps@headings{%
        \let\@oddfoot\@empty
        \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
        \let\@mkboth\markboth
        \def\sectionmark##1{%
          \markright {\MakeUppercase{%
            \ifnum \c@secnumdepth >\m@ne
              \thesection\quad
            \fi
            ##1}}}}
    \fi
    \def\ps@myheadings{%
        \let\@oddfoot\@empty\let\@evenfoot\@empty
        \def\@evenhead{\thepage\hfil\slshape\leftmark}%
        \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
        \let\@mkboth\@gobbletwo
        \let\sectionmark\@gobble
        \let\subsectionmark\@gobble
        }
    \setcounter{secnumdepth}{5}
    \newcounter {part}
    \newcounter {section}
    \newcounter {subsection}[section]
    \newcounter {subsubsection}[subsection]
    \newcounter {paragraph}[subsubsection]
    \newcounter {subparagraph}[paragraph]
    \renewcommand \thepart {\@Roman\c@part}
    \renewcommand \thesection {\@arabic\c@section}
    \renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
    \renewcommand\thesubsubsection{\thesubsection.\@arabic\c@subsubsection}
    \renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
    \renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
    \newcommand\part{%
       \if@noskipsec \leavevmode \fi
       \par
       \addvspace{4ex}%
       \@afterindentfalse
       \secdef\@part\@spart}

    \def\@part[#1]#2{%
        \ifnum \c@secnumdepth >\m@ne
          \refstepcounter{part}%
          \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
        \else
          \addcontentsline{toc}{part}{#1}%
        \fi
        {\parindent \z@ \raggedright
         \interlinepenalty \@M
         \normalfont
         \ifnum \c@secnumdepth >\m@ne
           \Large\bfseries \partname\nobreakspace\thepart
           \par\nobreak
         \fi
         \huge \bfseries #2%
         \markboth{}{}\par}%
        \nobreak
        \vskip 3ex
        \@afterheading}
    \def\@spart#1{%
        {\parindent \z@ \raggedright
         \interlinepenalty \@M
         \normalfont
         \huge \bfseries #1\par}%
         \nobreak
         \vskip 3ex
         \@afterheading}
    \newcommand\section{\@startsection {section}{1}{\z@}%
                                       {-3.5ex \@plus -1ex \@minus -.2ex}%
                                       {2.3ex \@plus.2ex}%
                                       {\normalfont\Large\bfseries}}
    \newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                         {-3.25ex\@plus -1ex \@minus -.2ex}%
                                         {1.5ex \@plus .2ex}%
                                         {\normalfont\large\bfseries}}
    \newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                         {-3.25ex\@plus -1ex \@minus -.2ex}%
                                         {1.5ex \@plus .2ex}%
                                         {\normalfont\normalsize\bfseries}}
    \newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                        {3.25ex \@plus1ex \@minus.2ex}%
                                        {-1em}%
                                        {\normalfont\normalsize\bfseries}}
    \newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                           {3.25ex \@plus1ex \@minus .2ex}%
                                           {-1em}%
                                          {\normalfont\normalsize\bfseries}}
    \if@twocolumn
      \setlength\leftmargini  {2em}
    \else
      \setlength\leftmargini  {2.5em}
    \fi
    \leftmargin  \leftmargini
    \setlength\leftmarginii  {2.2em}
    \setlength\leftmarginiii {1.87em}
    \setlength\leftmarginiv  {1.7em}
    \if@twocolumn
      \setlength\leftmarginv  {.5em}
      \setlength\leftmarginvi {.5em}
    \else
      \setlength\leftmarginv  {1em}
      \setlength\leftmarginvi {1em}
    \fi
    \setlength  \labelsep  {.5em}
    \setlength  \labelwidth{\leftmargini}
    \addtolength\labelwidth{-\labelsep}
    \@beginparpenalty -\@lowpenalty
    \@endparpenalty   -\@lowpenalty
    \@itempenalty     -\@lowpenalty
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
    \newcommand\labelitemi  {\labelitemfont \textbullet}
    \newcommand\labelitemii {\labelitemfont \bfseries \textendash}
    \newcommand\labelitemiii{\labelitemfont \textasteriskcentered}
    \newcommand\labelitemiv {\labelitemfont \textperiodcentered}
    \newcommand\labelitemfont{\normalfont}
    \newenvironment{description}
                   {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                            \let\makelabel\descriptionlabel}}
                   {\endlist}
    \newcommand*\descriptionlabel[1]{\hspace\labelsep
                                    \normalfont\bfseries #1}
    \newenvironment{verse}
                   {\let\\\@centercr
                    \list{}{\itemsep      \z@
                            \itemindent   -1.5em%
                            \listparindent\itemindent
                            \rightmargin  \leftmargin
                            \advance\leftmargin 1.5em}%
                    \item\relax}
                   {\endlist}
    \newenvironment{quotation}
                   {\list{}{\listparindent 1.5em%
                            \itemindent    \listparindent
                            \rightmargin   \leftmargin
                            \parsep        \z@ \@plus\p@}%
                    \item\relax}
                   {\endlist}
    \newenvironment{quote}
                   {\list{}{\rightmargin\leftmargin}%
                    \item\relax}
                   {\endlist}
    \newcommand\appendix{\par
      \setcounter{section}{0}%
      \setcounter{subsection}{0}%
      \gdef\thesection{\@Alph\c@section}}
    \setlength\arraycolsep{5\p@}
    \setlength\tabcolsep{6\p@}
    \setlength\arrayrulewidth{.4\p@}
    \setlength\doublerulesep{2\p@}
    \setlength\tabbingsep{\labelsep}
    \skip\@mpfootins = \skip\footins
    \setlength\fboxsep{3\p@}
    \setlength\fboxrule{.4\p@}
    \renewcommand \theequation {\@arabic\c@equation}
    \newcounter{figure}
    \renewcommand \thefigure {\@arabic\c@figure}
    \def\fps@figure{tbp}
    \def\ftype@figure{1}
    \def\ext@figure{lof}
    \def\fnum@figure{\figurename\nobreakspace\thefigure}
    \newenvironment{figure}
                   {\@float{figure}}
                   {\end@float}
    \newenvironment{figure*}
                   {\@dblfloat{figure}}
                   {\end@dblfloat}
    \newcounter{table}
    \renewcommand\thetable{\@arabic\c@table}
    \def\fps@table{tbp}
    \def\ftype@table{2}
    \def\ext@table{lot}
    \def\fnum@table{\tablename\nobreakspace\thetable}
    \newenvironment{table}
                   {\@float{table}}
                   {\end@float}
    \newenvironment{table*}
                   {\@dblfloat{table}}
                   {\end@dblfloat}
    \newlength\abovecaptionskip
    \newlength\belowcaptionskip
    \setlength\abovecaptionskip{10\p@}
    \setlength\belowcaptionskip{0\p@}
    \long\def\@makecaption#1#2{%
      \vskip\abovecaptionskip
      \sbox\@tempboxa{#1: #2}%
      \ifdim \wd\@tempboxa >\hsize
        #1: #2\par
      \else
        \global \@minipagefalse
        \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
      \fi
      \vskip\belowcaptionskip}
    \DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
    \DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
    \DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
    \DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
    \DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
    \DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
    \DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
    \DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
    \DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
    \newcommand*\l@part[2]{%
      \ifnum \c@tocdepth >-2\relax
        \addpenalty\@secpenalty
        \addvspace{2.25em \@plus\p@}%
        \setlength\@tempdima{3em}%
        \begingroup
          \parindent \z@ \rightskip \@pnumwidth
          \parfillskip -\@pnumwidth
          {\leavevmode
           \large \bfseries #1\hfil
           \hb@xt@\@pnumwidth{\hss #2%
                              \kern-\p@\kern\p@}}\par
           \nobreak
           \if@compatibility
             \global\@nobreaktrue
             \everypar{\global\@nobreakfalse\everypar{}}%
          \fi
        \endgroup
      \fi}
    \newcommand*\l@section[2]{%
      \ifnum \c@tocdepth >\z@
        \addpenalty\@secpenalty
        \addvspace{1.0em \@plus\p@}%
        \setlength\@tempdima{1.5em}%
        \begingroup
          \parindent \z@ \rightskip \@pnumwidth
          \parfillskip -\@pnumwidth
          \leavevmode \bfseries
          \advance\leftskip\@tempdima
          \hskip -\leftskip
          #1\nobreak\hfil
          \nobreak\hb@xt@\@pnumwidth{\hss #2%
                                     \kern-\p@\kern\p@}\par
        \endgroup
      \fi}
    \newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
    \newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
    \newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
    \newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
    \newdimen\bibindent
    \setlength\bibindent{1.5em}
    \newenvironment{thebibliography}[1]
         {\section*{\refname}%
          \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
          \list{\@biblabel{\@arabic\c@enumiv}}%
               {\settowidth\labelwidth{\@biblabel{#1}}%
                \leftmargin\labelwidth
                \advance\leftmargin\labelsep
                \@openbib@code
                \usecounter{enumiv}%
                \let\p@enumiv\@empty
                \renewcommand\theenumiv{\@arabic\c@enumiv}}%
          \sloppy
          \clubpenalty4000
          \@clubpenalty \clubpenalty
          \widowpenalty4000%
          \sfcode`\.\@m}
         {\def\@noitemerr
           {\@latex@warning{Empty `thebibliography' environment}}%
          \endlist}
    \newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
    \let\@openbib@code\@empty
    \newenvironment{theindex}
                   {\if@twocolumn
                      \@restonecolfalse
                    \else
                      \@restonecoltrue
                    \fi
                    \twocolumn[\section*{\indexname}]%
                    \@mkboth{\MakeUppercase\indexname}%
                            {\MakeUppercase\indexname}%
                    \thispagestyle{plain}\parindent\z@
                    \parskip\z@ \@plus .3\p@\relax
                    \columnseprule \z@
                    \columnsep 35\p@
                    \let\item\@idxitem}
                   {\if@restonecol\onecolumn\else\clearpage\fi}
    \newcommand\@idxitem{\par\hangindent 40\p@}
    \newcommand\subitem{\@idxitem \hspace*{20\p@}}
    \newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
    \newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
    \renewcommand\footnoterule{%
      \kern-3\p@
      \hrule\@width.4\columnwidth
      \kern2.6\p@}
    \newcommand\@makefntext[1]{%
        \parindent 1em%
        \noindent
        \hb@xt@1.8em{\hss\@makefnmark}#1}
    \newcommand\contentsname{Contents}
    \newcommand\listfigurename{List of Figures}
    \newcommand\listtablename{List of Tables}
    \newcommand\refname{References}
    \newcommand\indexname{Index}
    \newcommand\figurename{Figure}
    \newcommand\tablename{Table}
    \newcommand\partname{Part}
    \newcommand\appendixname{Appendix}
    \newcommand\abstractname{Abstract}
    \def\today{\ifcase\month\or
      January\or February\or March\or April\or May\or June\or
      July\or August\or September\or October\or November\or December\fi
      \space\number\day, \number\year}
    \setlength\columnsep{10\p@}
    \setlength\columnseprule{0\p@}
    \pagestyle{plain}
    \pagenumbering{arabic}
    \if@twoside
    \else
      \raggedbottom
    \fi
    \if@twocolumn
      \twocolumn
      \sloppy
      \flushbottom
    \else
      \onecolumn
    \fi
% End of file `article.cls'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ifxetex
  \RequirePackage{ifxetex}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Layout
    \RequirePackage{geometry}
    \newlength{\margintopbot}\setlength{\margintopbot}{0.75in}
    \newlength{\margingen}\setlength{\margingen}{1in}
    \newlength{\marginside}
    \ifbool{twocol}
        {\setlength{\marginside}{\margingen/2}\setlength{\columnsep}{\margingen/4}}
        {\setlength{\marginside}{\margingen}}
    \geometry{left=\marginside,right=\marginside,top=\margintopbot,bottom=\margintopbot}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{\baselineskip}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Colors
    \definecolor{DarkBlue}{HTML}{2D4CB9}
    \definecolor{LightBlue}{HTML}{E5EBFF}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Font
    \RequirePackage{anyfontsize}
    \ifxetex
        \RequirePackage{fontspec}
        % CohereText
            \newfontfamily{\CohereText}{CohereText}[Extension=.ttf,Path=\PATHCLASS/fonts/,
                UprightFont = *-Regular,
                ItalicFont = *-Italic,
                BoldFont = *-Bold,
                BoldItalicFont = *-BoldItalic,
                FontFace = {lt}{n}{*-Light},
                FontFace = {lt}{it}{*-LightItalic},
            ]
        % CohereHeadline
            \newfontfamily{\CohereHeadline}{CohereHeadline}[Extension=.ttf,Path=\PATHCLASS/fonts/,
                UprightFont = *-Regular,
                BoldFont = *-Bold,
                FontFace = {lt}{n}{*-Light},
            ]
        %
        \DeclareRobustCommand{\ltseries}{\fontseries{lt}\selectfont}
        \DeclareTextFontCommand{\textlt}{\ltseries}
    \else
        \RequirePackage[T1]{fontenc}
        \RequirePackage[scaled]{berasans}
        \DeclareRobustCommand{\ltseries}{}
        \DeclareTextFontCommand{\textlt}{}
        \def\CohereText{\sffamily}
        \def\CohereHeadline{\sffamily}
    \fi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cohere logo
    \newcommand{\ChoereLogo}[2]{\resizebox{#1}{!}{%
        \begin{tikzpicture}[x=1pt,y=-1pt,inner sep=0pt,outer sep=0pt]
            \fill[#2] (380.6929,145.9843) .. controls (380.6929, 54.4252) and (425.1969, 1.1339) .. (502.5827, 1.1339) .. controls (579.9685, 1.1339) and (599.2441, 36.2835) .. (610.0157, 96.9449) -- (572.5984,96.9449) .. controls (562.9606, 54.9921) and (538.8661, 33.7323) .. (502.0157, 33.7323) .. controls (447.3071, 33.7323) and (416.9764, 74.8346) .. (416.9764, 146.2677) .. controls (416.9764, 217.7008) and (446.1732, 258.8031) .. (500.8819, 258.8031) .. controls (555.5906, 258.8031) and (562.1102, 238.1102) .. (572.3150, 195.5906) -- (609.7323,195.5906) .. controls (598.6772, 256.8189) and (559.2756, 291.4016) .. (500.8819, 291.4016) .. controls (442.4882, 291.4016) and (380.6929, 237.5433) .. (380.6929, 146.2677) -- cycle;
            \fill[#2] (643.4646,184.8189) .. controls (643.4646, 121.6063) and (685.7008, 78.5197) .. (746.6457, 78.5197) .. controls (807.5906, 78.5197) and (849.8268, 121.6063) .. (849.8268, 184.8189) .. controls (849.8268, 248.0315) and (807.5906, 291.1181) .. (746.6457, 291.1181) .. controls (685.7008, 291.1181) and (643.4646, 248.0315) .. (643.4646, 184.8189) -- cycle(815.5276,184.8189) .. controls (815.5276, 140.3150) and (787.7480, 108.2835) .. (746.6457, 108.2835) .. controls (705.5433, 108.2835) and (677.7638, 140.5984) .. (677.7638, 184.8189) .. controls (677.7638, 229.0394) and (705.5433, 261.3543) .. (746.6457, 261.3543) .. controls (787.7480, 261.3543) and (815.5276, 229.0394) .. (815.5276, 184.8189) -- cycle;
            \fill[#2] (1075.4646,168.9449) -- (1075.4646,288.8504) -- (1042.0157,288.8504) -- (1042.0157,171.2126) .. controls (1042.0157, 130.9606) and (1022.1732, 109.4173) .. (987.0236, 109.4173) .. controls (951.8740, 109.4173) and (925.7953, 137.1969) .. (925.7953, 181.1339) -- (925.7953,289.1339) -- (892.3465,289.1339) -- (892.3465,3.4016) -- (925.7953,3.4016) -- (925.7953,109.7008) .. controls (940.5354, 89.8583) and (964.0630, 78.2362) .. (992.6929, 78.2362) .. controls (1043.7165, 78.2362) and (1075.4646, 109.1339) .. (1075.4646, 168.6614) -- cycle;
            \fill[#2] (1151.1496,192.1890) -- (1151.1496,194.1732) .. controls (1152.2835, 233.5748) and (1177.2283, 261.3543) .. (1216.6299, 261.3543) .. controls (1256.0315, 261.3543) and (1268.7874, 245.7638) .. (1275.5906, 219.1181) -- (1309.3228,219.1181) .. controls (1301.1024, 260.7874) and (1267.0866, 291.1181) .. (1218.8976, 291.1181) .. controls (1170.7087, 291.1181) and (1115.7165, 248.3150) .. (1115.7165, 184.8189) .. controls (1115.7165, 121.3228) and (1155.6850, 78.5197) .. (1214.3622, 78.5197) .. controls (1273.0394, 78.5197) and (1311.3071, 115.6535) .. (1313.0079, 177.1654) .. controls (1313.0079, 180.2835) and (1312.7244, 187.3701) .. (1312.1575, 192.1890) -- (1150.8661,192.1890) -- cycle(1153.1339,163.5591) -- (1277.2913,163.5591) .. controls (1274.1732, 127.8425) and (1249.5118, 107.7165) .. (1214.9291, 107.7165) .. controls (1180.3465, 107.7165) and (1156.2520, 129.8268) .. (1153.1339, 163.5591) -- cycle;
            \fill[#2] (1513.7008,80.7874) -- (1513.7008,113.3858) -- (1486.7717,113.3858) .. controls (1438.2992, 113.3858) and (1424.1260, 150.5197) .. (1424.1260, 183.4016) -- (1424.1260,257.9528) -- (1484.7874,257.9528) -- (1484.7874,288.5669) -- (1344.1890,288.5669) -- (1344.1890,257.9528) -- (1390.6772,257.9528) -- (1390.6772,111.4016) -- (1344.1890,111.4016) -- (1344.1890,80.7874) -- (1422.4252,80.7874) -- (1423.5591,115.9370) .. controls (1432.9134, 99.4961) and (1451.6220, 80.7874) .. (1491.3071, 80.7874) -- (1513.7008,80.7874) -- cycle;
            \fill[#2] (1573.2283,192.1890) -- (1573.2283,194.1732) .. controls (1574.3622, 233.5748) and (1599.3071, 261.3543) .. (1638.7087, 261.3543) .. controls (1678.1102, 261.3543) and (1690.8661, 245.7638) .. (1697.6693, 219.1181) -- (1731.4016,219.1181) .. controls (1723.1811, 260.7874) and (1689.1654, 291.1181) .. (1640.9764, 291.1181) .. controls (1592.7874, 291.1181) and (1537.7953, 248.3150) .. (1537.7953, 184.8189) .. controls (1537.7953, 121.3228) and (1577.7638, 78.5197) .. (1636.4409, 78.5197) .. controls (1695.1181, 78.5197) and (1733.3858, 115.6535) .. (1735.0866, 177.1654) .. controls (1735.0866, 180.2835) and (1734.8031, 187.3701) .. (1734.2362, 192.1890) -- (1572.9449,192.1890) -- cycle(1575.4961,163.5591) -- (1699.6535,163.5591) .. controls (1696.2520, 127.8425) and (1671.8740, 107.7165) .. (1637.2913, 107.7165) .. controls (1602.7087, 107.7165) and (1578.6142, 129.8268) .. (1575.4961, 163.5591) -- cycle;
            \fill[#2] (1910.2677,288.5669) -- (1910.2677,3.4016) -- (1945.7008,3.4016) -- (1945.7008,255.9685) -- (2073.5433,255.9685) -- (2073.5433,288.5669) -- (1910.2677,288.5669) -- cycle;
            \fill[#2] (2296.6299,258.2362) -- (2296.6299,288.8504) -- (2278.2047,288.8504) .. controls (2250.4252, 288.8504) and (2241.0709, 276.9449) .. (2240.7874, 256.5354) .. controls (2227.7480, 275.2441) and (2207.0551, 291.1181) .. (2171.0551, 291.1181) .. controls (2135.0551, 291.1181) and (2094.5197, 268.4409) .. (2094.5197, 230.4567) .. controls (2094.5197, 192.4724) and (2123.4331, 165.5433) .. (2178.1417, 165.5433) -- (2239.3701,165.5433) -- (2239.3701,151.3701) .. controls (2239.3701, 124.4409) and (2220.0945, 108.2835) .. (2187.4961, 108.2835) .. controls (2154.8976, 108.2835) and (2138.7402, 122.1732) .. (2134.4882, 143.4331) -- (2101.0394,143.4331) .. controls (2105.8583, 102.6142) and (2139.0236, 78.5197) .. (2188.9134, 78.5197) .. controls (2238.8031, 78.5197) and (2272.5354, 104.8819) .. (2272.5354, 153.3543) -- (2272.5354,244.3465) .. controls (2272.5354, 255.4016) and (2276.5039, 258.2362) .. (2285.8583, 258.2362) -- (2296.3465,258.2362) -- cycle(2239.0866,193.8898) -- (2174.7402,193.8898) .. controls (2144.9764, 193.8898) and (2128.2520, 204.9449) .. (2128.2520, 228.4724) .. controls (2128.2520, 252.0000) and (2145.8268, 262.7717) .. (2173.6063, 262.7717) .. controls (2215.2756, 262.7717) and (2239.0866, 238.6772) .. (2239.0866, 204.0945) -- (2239.0866,193.8898) -- cycle;
            \fill[#2] (2536.4409,184.8189) .. controls (2536.4409, 248.3150) and (2494.2047, 291.1181) .. (2436.6614, 291.1181) .. controls (2379.1181, 291.1181) and (2377.9843, 276.9449) .. (2365.2283, 256.8189) -- (2360.6929,288.5669) -- (2331.7795,288.5669) -- (2331.7795,3.4016) -- (2365.2283,3.4016) -- (2365.2283,114.2362) .. controls (2379.1181, 94.9606) and (2401.5118, 78.5197) .. (2436.6614, 78.5197) .. controls (2494.2047, 78.5197) and (2536.4409, 117.3543) .. (2536.4409, 184.8189) -- cycle(2502.1417,184.8189) .. controls (2502.1417, 139.4646) and (2474.3622, 108.2835) .. (2433.2598, 108.2835) .. controls (2392.1575, 108.2835) and (2364.9449, 139.7480) .. (2364.9449, 183.9685) .. controls (2364.9449, 228.1890) and (2392.7244, 261.3543) .. (2433.2598, 261.3543) .. controls (2473.7953, 261.3543) and (2502.1417, 229.8898) .. (2502.1417, 184.8189) -- cycle;
            \fill[#2] (2566.7717,220.2520) -- (2601.0709,220.2520) .. controls (2602.2047, 244.6299) and (2623.7480, 262.4882) .. (2658.6142, 262.4882) .. controls (2693.4803, 262.4882) and (2708.7874, 249.7323) .. (2708.7874, 230.7402) .. controls (2708.7874, 211.7480) and (2686.1102, 202.6772) .. (2654.9291, 198.9921) .. controls (2608.4409, 193.3228) and (2571.3071, 183.9685) .. (2571.3071, 140.5984) .. controls (2571.3071, 97.2283) and (2605.8898, 78.2362) .. (2652.6614, 78.2362) .. controls (2699.4331, 78.2362) and (2734.0157, 100.6299) .. (2737.4173, 143.7165) -- (2703.1181,143.7165) .. controls (2700.5669, 122.4567) and (2681.0079, 106.5827) .. (2652.6614, 106.5827) .. controls (2624.3150, 106.5827) and (2604.1890, 118.7717) .. (2604.1890, 138.0472) .. controls (2604.1890, 157.3228) and (2626.2992, 163.8425) .. (2656.3465, 167.2441) .. controls (2703.6850, 172.9134) and (2741.3858, 181.9843) .. (2741.3858, 227.9055) .. controls (2741.3858, 273.8268) and (2704.2520, 290.5512) .. (2658.3307, 290.5512) .. controls (2612.4094, 290.5512) and (2567.3386, 265.3228) .. (2566.7717, 219.6850) -- cycle;
            \fill[#2] (213.1654,46.2047) -- (237.8268,22.3937) .. controls (243.2126, 17.2913) and (251.7165, 17.2913) .. (256.8189, 22.6772) .. controls (261.9213, 28.0630) and (261.9213, 36.5669) .. (256.5354, 41.6693) -- (217.1339,79.6535) .. controls (182.2677, 113.3858) and (183.1181, 169.7953) .. (219.4016, 202.3937) -- (259.9370,238.9606) .. controls (265.3228, 244.0630) and (265.8898, 252.5669) .. (260.7874, 257.9528) .. controls (255.6850, 263.3386) and (247.1811, 263.9055) .. (241.7953, 258.8031) -- (202.6772,223.3701) -- (213.7323,275.2441) .. controls (215.4331, 282.6142) and (210.6142, 289.7008) .. (203.2441, 291.1181) .. controls (195.8740, 292.8189) and (188.7874, 288.0000) .. (187.3701, 280.6299) -- (176.0315,227.0551) .. controls (165.8268, 179.4331) and (117.6378, 150.5197) .. (70.8661, 163.8425) -- (18.1417,178.8661) .. controls (11.0551, 180.8504) and (3.4016, 176.8819) .. (1.4173, 169.5118) .. controls (-0.5669, 162.4252) and (3.4016, 154.7717) .. (10.7717, 152.7874) -- (44.7874,143.1496) -- (14.7402,134.0787) .. controls (7.6535, 131.8110) and (3.6850, 124.4409) .. (5.6693, 117.3543) .. controls (7.9370, 110.2677) and (15.3071, 106.2992) .. (22.3937, 108.2835) -- (74.8346,124.1575) .. controls (121.3228, 138.3307) and (170.0787, 110.2677) .. (181.1339, 62.6457) -- (193.6063,9.3543) .. controls (195.3071, 2.2677) and (202.3937, -2.2677) .. (209.7638, -0.5669) .. controls (217.1339, 1.1339) and (221.3858, 8.2205) .. (219.9685, 15.5906) -- (213.1654,45.3543) -- (213.1654,45.3543) -- cycle(166.6772,161.8583) .. controls (164.9764, 152.7874) and (164.1260, 143.7165) .. (164.6929, 134.6457) .. controls (158.1732, 139.4646) and (151.0866, 143.4331) .. (144.0000, 146.5512) .. controls (152.2205, 150.8031) and (159.8740, 155.9055) .. (166.6772, 161.8583) -- cycle;
        \end{tikzpicture}}
    }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Language
    \ifxetex
        \RequirePackage{polyglossia}
        \setdefaultlanguage{english}
    \else
        \RequirePackage[english]{babel}
    \fi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Math
    \RequirePackage{amsmath}
    \RequirePackage{amsfonts}
    \RequirePackage{amssymb}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lists
    \RequirePackage{enumitem}
    %\setlist{leftmargin=*,nosep,noitemsep}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tables
    \RequirePackage{booktabs}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figures
    \RequirePackage{graphicx}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Caption
    \RequirePackage{caption}
    \RequirePackage{subcaption}
    \captionsetup{font=footnotesize}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Boxes
    \RequirePackage[most]{tcolorbox}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Headers
    \RequirePackage{fancyhdr}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancypagestyle{coherefirst}{\fancyhf{}
        \fancyfoot[C]{\small\thepage}
        \fancyfoot[L]{\small Released as a preprint on~\@date}
    }
    \fancypagestyle{cohere}{\fancyhf{}
        \fancyfoot[C]{\small\thepage}
    }
    \pagestyle{cohere}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Watermark
    \ifbool{cdraft}{%
        \RequirePackage{watermark}
        \watermark{\tikz[remember picture,overlay]{\node[opacity=0.25,rotate={atan(\paperheight/\paperwidth)},scale=10,font=\bfseries] at (current page.center) {Draft Not Final};}}
    }{}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sectioning
    \RequirePackage{titlesec}
    %
    \def\ALsec{\raggedright}
    \def\FONTsec{\CohereText\Large\color{DarkBlue}}
    \def\FONTssec{\CohereText\large\color{DarkBlue}}
    \def\FONTsssec{\CohereText\color{DarkBlue}}
    \def\FONTpar{\CohereText\itshape\color{DarkBlue}}
    \def\FONTspar{\CohereText\color{DarkBlue}}
    %
    \titleformat{\section}[hang]{\ALsec\FONTsec}{\thesection}{1em}{}[]
    \titleformat{\subsection}[hang]{\ALsec\FONTssec}{\thesubsection}{1em}{}[]
    \titleformat{\subsubsection}[hang]{\ALsec\FONTsssec}{\thesubsubsection}{1em}{}[]
    \titleformat{\paragraph}[hang]{\ALsec\FONTpar}{\theparagraph}{1em}{}[]
    \titleformat{\subparagraph}[hang]{\ALsec\FONTspar}{\thesubparagraph}{1em}{}[]
    %
    \titleformat{name=\section,numberless}[hang]{\ALsec\FONTsec}{}{0pt}{}[]
    \titleformat{name=\subsection,numberless}[hang]{\ALsec\FONTssec}{}{0pt}{}[]
    \titleformat{name=\subsubsection,numberless}[hang]{\ALsec\FONTssssec}{}{0pt}{}[]
    \titleformat{name=\paragraph,numberless}[hang]{\ALsecFONTpar}{}{0pt}{}[]
    \titleformat{name=\subparagraph,numberless}[hang]{\ALsecFONTspar}{}{0pt}{}[]
    %
    \titlespacing{\section}{0pt}{0.5\baselineskip}{-0.25\baselineskip}
    \titlespacing{\subsection}{0pt}{0.5\baselineskip}{-0.25\baselineskip}
    \titlespacing{\subsubsection}{0pt}{0.5\baselineskip}{-0.25\baselineskip}
    \titlespacing{\paragraph}{0pt}{0.5\baselineskip}{-0.25\baselineskip}
    \titlespacing{\subparagraph}{0pt}{0.5\baselineskip}{-0.25\baselineskip}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section findings
    \NewDocumentEnvironment{sectionfindings}{ +b }{\begin{tcolorbox}[title={Section Findings},
            fonttitle=\CohereText\fontsize{12pt}{12pt}\color{white}\selectfont,
            enhanced,rounded corners,arc=4pt,
            colback=LightBlue,
            colbacktitle=DarkBlue,
            colframe=black,
            boxsep=0pt,
            leftrule=0.2pt,rightrule=0.2pt,
            toprule=0.2pt,bottomrule=0.2pt,
            titlerule=0.2pt,
            left=10pt,right=10pt,
            top=10pt,bottom=10pt,
            toptitle=4pt,bottomtitle=4pt,
            halign=flush left,]
        \begin{itemize}[leftmargin=*]
            #1
        \end{itemize}
    \end{tcolorbox}}{}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliography
    \RequirePackage{natbib}
    \setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}
    \bibliographystyle{\PATHCLASS/cohere}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors
    \newcounter{authorscount}
    \newcounter{authc}
    \makeatletter
        \define@key{choauth}{name}{\csdef{Aname\theauthorscount}{#1}}
        \define@key{choauth}{affiliation}{\csdef{Aaffi\theauthorscount}{#1}}
        \define@key{choauth}{email}{\csdef{Aemail\theauthorscount}{#1}}
    \makeatother
    \setkeys{choauth}{}
    \renewcommand{\author}[1]{\stepcounter{authorscount}%
        \csdef{Aname\theauthorscount}{}%
        \csdef{Aaffi\theauthorscount}{}%
        \csdef{Aemail\theauthorscount}{}%
        \setkeys{choauth}{#1}%
    }
    \def\dbaffiliation{}\newcommand{\affiliation}[1]{\def\dbaffiliation{#1}}
    \def\dbemail{}\newcommand{\email}[1]{\def\dbemail{#1}}
    \def\dbaffiliations{}\newcommand{\affiliations}[1]{\def\dbaffiliations{#1}}
    \def\dbcorresponding{}\newcommand{\corresponding}[2][]{\def\dbcorrP{#1}\def\dbcorresponding{#2}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title
    \newcounter{fa}\setcounter{fa}{0}
    \newcounter{psa}\setcounter{psa}{0}
    \newcounter{aff}\setcounter{aff}{0}
    \newcommand{\fasymbol}{\textcolor{DarkBlue}{$\bigstar$}}
    \newcommand{\fa}{\stepcounter{fa}\textsuperscript{\fasymbol}}
    \newcommand{\psasymbol}{\textcolor{DarkBlue}{$\blacklozenge$}}
    \newcommand{\psa}{\stepcounter{psa}\textsuperscript{\psasymbol}}
    %
    \newcommand{\multiauthors}{\booltrue{maut}}
    %
    \newcommand{\FONTauthorblock}{}
    \newcommand{\FONTauthor}{\small\strut}
    \newcommand{\FONTauthors}{\small}
    \newcommand{\FONTaff}{\itshape\strut}
    \newcommand{\FONTemail}{\ttfamily\strut}
    \def\dbtitle{}\renewcommand{\title}[1]{\def\dbtitle{#1}}
    \def\dbsubtitle{}\newcommand{\subtitle}[1]{\def\dbsubtitle{#1}}
    % Title box
        \newcommand{\FONTtitlepageTitle}{\CohereHeadline\fontsize{25pt}{30pt}\color{DarkBlue}\selectfont}
        \newcommand{\FONTtitlepageSubtitle}{\CohereHeadline\fontsize{20pt}{24pt}\color{DarkBlue}\selectfont}
        \newtcolorbox{titlebox}{enhanced,sharp corners,frame hidden,
            colback=white,
            fontupper=\FONTtitlepageTitle,
            boxsep=0pt,leftrule=0pt,rightrule=0pt,
            toprule=0pt,bottomrule=0pt,
            left=0pt,right=0pt,top=23pt,bottom=23pt,
            halign=flush left,
            underlay={
                \draw[DarkBlue,line width=0.8pt] (frame.north west) -- (frame.north east)
                    node[at start,above right,inner sep=0pt,outer sep=0pt,yshift=7.5pt] {\ChoereLogo{75pt}{DarkBlue}}
                ;
                \draw[DarkBlue,line width=0.8pt] (frame.south west) -- (frame.south east);
            }
        }
    % Abstract box
        \newcommand{\FONTtitlepageAbstracttitle}{\CohereText\fontsize{16pt}{19.2pt}\color{DarkBlue}\selectfont}
        \newcommand{\FONTtitlepageAbstract}{\fontsize{9.5pt}{14pt}\selectfont}
        \newtcolorbox{abstractbox}{
            enhanced,rounded corners,arc=4pt,
            colback=LightBlue,
            colframe=black,
            fontupper=\FONTtitlepageAbstract,
            boxsep=0pt,
            leftrule=0.2pt,rightrule=0.2pt,
            toprule=0.2pt,bottomrule=0.2pt,
            left=15pt,right=15pt,
            top=15pt,bottom=15pt,
            halign=flush left,
            before=\vspace{-20pt}
        }
        \newlength{\authorspacefont}
    % Print title
    \newcommand{\PrintTitle}{
        % Title
            \begin{titlebox}
                \FONTtitlepageTitle\dbtitle\ifcsstring{dbsubtitle}{}{}{\\{\FONTtitlepageSubtitle\dbsubtitle}}%
            \end{titlebox}
        % % Authors
            \ifbool{maut}%
            % Multiple authors
                {
                    \ifbool{twocol}{\vspace{0.5\baselineskip}}{}
                    \begin{flushleft}\FONTauthors
                        \foreach \auth in {1,...,\theauthorscount} {%
                            \ifnumequal{\auth}{\theauthorscount}{and~}{}%
                            \mbox{\csuse{Aname\auth}}%
                            \ifcsstring{Aaffi\auth}{}{}{\textsuperscript{\csuse{Aaffi\auth}}}%
                            \ifnumequal{\auth}{\theauthorscount}{}{, }%
                        }\par%
                    \end{flushleft}%
                    \ifcsstring{dbaffiliations}{}{}{
                        \ifbool{twocol}{}{\vspace{-1.5\baselineskip}}
                        \begin{flushleft}\footnotesize\setlength{\authorspacefont}{\the\fontdimen2\font}
                            \renewcommand{\item}[1][]{\stepcounter{aff}\ifnumequal{\theaff}{1}{}{~\kern-2\authorspacefont, } \textsuperscript{##1}\kern-\the\fontdimen3\font\kern1pt}%
                            \dbaffiliations\par%
                        \end{flushleft}
                    }%
                    \ifcsstring{dbcorresponding}{}{}{%
                        \ifbool{twocol}{}{\vspace{-1.5\baselineskip}}
                        \begin{flushleft}\footnotesize
                            Corresponding author\ifcsstring{dbcorrP}{*}{s}{}: \dbcorresponding\par
                        \end{flushleft}
                    }%
                }% end
            % Regular authors style
                {%
                    \ifnumequal{\theauthorscount}{1}{\renewcommand{\FONTauthorblock}{\LARGE}}{}%
                    \ifnumequal{\theauthorscount}{2}{\renewcommand{\FONTauthorblock}{\Large}}{}%
                    \vspace{1.5\baselineskip}
                    \ifcsstring{dbaffiliation}{}{%
                        % No affiliation
                        \ifnumgreater{\theauthorscount}{2}{\def\AuthCol{3}}{\def\AuthCol{\theauthorscount}}
                        \foreach \auth in {1,...,\theauthorscount} {\stepcounter{authc}%
                            \ifnumequal{\theauthc}{1}{\strut\hfill}{}%
                            \begin{minipage}[t]{\linewidth/\AuthCol-10pt}\centering\FONTauthorblock
                                {\FONTauthor\FONTauthorblock\strut\csuse{Aname\auth}}\\
                                \ifcsstring{Aaffi\auth}{}{}{{\FONTaff\strut\csuse{Aaffi\auth}}\\}
                                \ifcsstring{Aemail\auth}{}{}{{\FONTemail\strut\csuse{Aemail\auth}}\\}
                            \end{minipage}%
                            \hfill\strut\ifnumequal{\theauthc}{3}{\setcounter{authc}{0}\\[\baselineskip]}{}
                        }%
                    }%
                    {%
                        % Affiliation
                        \begin{minipage}[t]{\linewidth}\centering\FONTauthorblock
                            \foreach \auth in {1,...,\theauthorscount} {
                                {\FONTauthor\FONTauthorblock\csuse{Aname\auth}\ifnumequal{\auth}{\theauthorscount}{}{,~}}%
                            }\\
                            {\FONTaff\dbaffiliation}\\
                            {\FONTemail\{\foreach \auth in {1,...,\theauthorscount} {\csuse{Aemail\auth}\ifnumequal{\auth}{\theauthorscount}{}{,}}\}@\dbemail}
                        \end{minipage}%
                    }
                    % corresponding
                    \ifcsstring{dbcorresponding}{}{}{%
                        
                        \vspace{0.5\baselineskip}
                        \ifbool{twocol}{\begin{adjustwidth}{0.1\linewidth}{0.1\linewidth}}{\begin{minipage}{\linewidth}}
                        \begin{flushleft}\footnotesize
                            Corresponding author\ifcsstring{dbcorrP}{*}{s}{}: \dbcorresponding\par
                        \end{flushleft}
                        \ifbool{twocol}{\end{adjustwidth}}{\end{minipage}\par}
                    }%
                }% end regular authors style

        % leave the previous line blank
        % Abstract
            \vspace{\baselineskip}
            \ifcsstring{abstractcont}{}{}{\begin{abstractbox}
                {\FONTtitlepageAbstracttitle\abstractname}\\
                \abstractcont
            \end{abstractbox}}

        % leave the previous line blank
        % Paper
            \ifbool{twocol}{\vspace{1.5\baselineskip}}{\vspace{\baselineskip}}
    }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Abstract
    \def\abstractcont{}
    \newcommand{\abstract}[1]{\def\abstractcont{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% At end preamble
    %\newcommand{\dbauthors}{}
    %\AtEndPreamble{\foreach \auth in {1,...,\theauthorscount} {\gappto\dbauthors{\csuse{Aname\auth} }}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% At begin document
\AfterEndPreamble{\thispagestyle{coherefirst}
        \ifbool{twocol}{\twocolumn[\PrintTitle]}{\PrintTitle}%
        % First authors footnote
            \ifnumequal{\thefa}{0}{}{\begingroup\renewcommand*{\thefootnote}{}\footnotetext{\textsuperscript{\fasymbol}First author\ifnumequal{\thefa}{1}{}{s}.}\endgroup}%
        % Principal senior advisor footnote
            \ifnumequal{\thepsa}{0}{}{\begingroup\renewcommand*{\thefootnote}{}\footnotetext{\textsuperscript{\psasymbol}Principal senior advisor\ifnumequal{\thepsa}{1}{}{s}.}\endgroup}%
    }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hyperref
    \RequirePackage{xurl}
    \RequirePackage{hyperref}
    \hypersetup{
        pdftitle={\dbtitle},
        pdfauthor={Cohere},
        pdfsubject={Cohere preprint},
        pdfkeywords={Cohere},
        pdfcreator={Cohere},
        pdfproducer={Cohere},
        colorlinks=true,
        linkcolor=DarkBlue,
        citecolor=DarkBlue,
        urlcolor=DarkBlue,
        filecolor=DarkBlue
    }
    \RequirePackage{cleveref}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\endinput
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%