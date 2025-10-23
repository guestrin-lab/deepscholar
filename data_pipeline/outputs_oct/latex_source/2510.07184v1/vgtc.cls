% `vgtc' LaTeX class. 
% THIS FILE MAY NOT BE MODIFIED WITHOUT THE EXPRESS APPROVAL
% OF THE VGTC PUBLICATIONS CHAIR (submissions@vgtc.org).
%
% Changelog:
% - modifications by Matthew Brehmer on 2024/02/12
%   - ACM CCS keywords replaced with index terms, so as to be consistent with full papers
%   - added cleveref package and instructions for use, rather than using autoref (mirrorring full paper template)
%     - added Supplemental Materials and Figure Credits sections at the end.
%     - added supplemental materials instructions and goals, an example project 
%       on OSF.
%     - added figure credits for CypressView.
%     - replaced paper count figure with newly-made ones.
%     - changed cross references to point to the top of the target float, rather than the caption.
%     - added blue hyperlink coloring, now that electronic has long been the default.
%     - removed sample.pdf & sample.eps.
%     - removed Google Scholar fields from template.bib.

% - modifications by Cagatay Turkay and Soumya Dutta on 2019/09/11
%   - modified the preprint article style to contain the IEEE copyright notices to produce IEEE compliant OA preprints.
%   - included two different statements for journal and conference papers
%   - added a new command \ieeedoi for authors to set IEEE generated DOIs on their preprints
%   - added comments on the template.tex for preprint guidance
% - modifications by Tobias Isenberg on 2017/07/05
%   - updated the instructions for author footer in template.tex for the journal template
% - modifications by Tobias Isenberg on 2017/06/14
%   - improved handling of marginpars for conference style, added examples for both styles
% - modifications by Tobias Isenberg on 2017/06/13
%   - updated link to ACM CCS in conference template tex
%   - transitioned to the updated 2012 system to be used with \CCScatTwelve (1998 still possible)
%   - updated CCS examples in conference template tex
%   - added statement in journal template tex to leave the copyright block untouched
%   - updated the DOI links in the bst files to use shorter https version
% - modifications by Filip Sadlo/Tobias Isenberg on 2017/03/16
%   - revived latex mode (as opposed to pdflatex) by disabling \let\ifpdf\relax
%   - note that packages that define \ifpdf (and thus lead to already defined
%     \ifpdf) are obsolete and should be changed to use ifpdf package instead
%   - added a comment to the actual tex template for those with compilation errors
% - modifications by Tobias Isenberg on 2016/07/19
%   - fixed a bug in the hyperref data (missing comma, thanks to Christian
%     Tominski)
%   - adjusted behavior for when \vgtccategory{} is not set in review mode (print
%     out "n/a")
%   - added switch to print out paper type in review mode if desired in review
%     mode, does not print anything if not used
%   - added teaser width computation, limited the entire teaser to the width of
%     the abstract
%   - fixes to the bibTeX templates for how addresses in proceedings are used,
%     now similar to inproceeding
%   - fixes to the bibTeX templates for dois with more than one dash in them
%   - fixes to the bibTeX templates: dois are now linked with https if only the
%     pure doi was given
%   - fixed caption behavior for tables due to caption expected above the table
% - modifications by Filip Sadlo on 2016/03/16
%   - replaced \pdfoutput mechanism by \ifpdf because \pdfoutput was
%     always detecting pdf output mode on current texlive installations
%   - The modification follows the current version of egpubl.cls
% - modifications by Tobias Isenberg on 2016/03/24
%   - improved marginpar
% - modifications by Tobias Isenberg on 2015/03/24
%   - better compilation for pdfLaTeX
%   - electronic option is now mandatory
%   - hyperref settings redone, information automatically used for pdf information
%   - microtype
%   - some suggestions for style use
%   - alternative bst files with DOI printing and hyperlinking
% - modifications by Meghan Haley on 2011/03/10
%   - manuscript received dates changed for 2011 journal style
% - modifications by Steven Bergner on 2009/03/28
%   - revived teaser feature
%   - added captionmargin to align teaser caption with abstract width
% - modifications by Meghan Haley on 2006/07/20
%   - manuscript received changed to 31 March for journal style
% - modifications by Steven Bergner on 2006/06/28
%   - made review and preprint work for journal style
%   - leveraged \firstsection{..} title above double column text
%   - included diamond line (currently by loading .eps file)
%   - adjusted headlines and footer (special on first page, with copyrighttext)
% - modifications by Steven Bergner on 2006/05/21
%   - included 'electronic' option using method from egpubl.cls (c)by D.Fellner
%   - included double column (full width) abstract/keywords/index terms,
%     which can be toggled by (ex/in)clusion of 'journal' document option
%     note: abstract is now a command not an environment, see template.tex
%   - copyrightspace enabled by default with opt. \nocopyrightspace switch
%   - 'CR categories' now called 'Index Terms'
%   - adjusted {sub|subsub|}section headline spacing
%   - adjusted to vgtc naming (from tcvg or acm)
% - modification of the acmsiggraph.cls class
% - modifications on 2004/04/06 by Torsten Moeller
%    * many modifications to conform to VGTC standard
%      - new heading style
%      - new caption style
%      - new reference style
%      - copyrightspace reduced to 0.5in
% - borrows *heavily* from Daniel Kartch's 'siggraph' class. 
% - also uses pieces of 'apalike.sty' and 'authordate1-4.cls'
% - version 0.1 2001/06/01 Stephen Spencer (spencer@acm.org)
% - version 0.2 2001/10/15 Stephen Spencer
%   - the "\onlineid" variable works with the "review" mode, placing a
%     banner across the top of each page "Online Submission ID 'onlineid'
% - version 0.3 2002/01/11 Stephen Spencer
%   - disabled the cover page option.
% - version 0.4 2002/01/23 Stephen Spencer
%   - based on suggestions from James O'Brien, the following changes:
%     - correction if '\ifcamera' and '\ifcameraelse' commands
%     - page numbering in both review and preprint modes
%     have been made.

% ------------ identification --------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{vgtc}[2006/05/21 IEEE VGTC]

% ------------ initial code --------------

\newif\ifvgtc@camera
\newif\ifvgtc@preprint
\newif\ifvgtc@review
\newif\ifvgtc@doublespaced
\newif\ifvgtc@wide                    \vgtc@widefalse
\newif\ifhavecopyrightspace           \havecopyrightspacetrue
\newif\ifvgtcjournal                  \vgtcjournalfalse
\newif\iftvcgsize                     \tvcgsizefalse

\newcommand{\vgtc@columnmode}{}
\newcommand{\vgtc@pointsize}{}

\RequirePackage{ifpdf}%

%% These few lines make a distinction between latex and pdflatex calls and they
%% bring in essential packages for graphics and font handling.
\ifpdf%                                % if we use pdflatex
  \pdfoutput=1\relax                   % create PDFs from pdfLaTeX
  \pdfcompresslevel=9                  % PDF Compression
  \pdfoptionpdfminorversion=7          % create PDF 1.7
  \ExecuteOptions{pdftex}
  \RequirePackage{graphicx}                % allow us to embed graphics files
  \DeclareGraphicsExtensions{.pdf,.png,.jpg,.jpeg} % for pdflatex we expect .pdf, .png, or .jpg files
\else%                                 % else we use pure latex
  \ExecuteOptions{dvips}
  \RequirePackage{graphicx}                % allow us to embed graphics files
  \DeclareGraphicsExtensions{.eps}     % for pure latex we expect eps files
\fi%

% ------------ declaration of options --------------

% 'cameraready' option.

\DeclareOption{cameraready}{%
   \vgtc@cameratrue%
   \vgtc@preprintfalse%
   \vgtc@reviewfalse%
   \renewcommand{\vgtc@columnmode}{twocolumn}%
   \vgtc@doublespacedfalse%
   \renewcommand{\vgtc@pointsize}{9pt}}

% 'review' option.

\DeclareOption{review}{%
   \vgtc@camerafalse%
   \vgtc@preprintfalse%
   \vgtc@reviewtrue%
   \renewcommand{\vgtc@columnmode}{twocolumn}%
   \vgtc@doublespacedfalse%
   \renewcommand{\vgtc@pointsize}{9pt}
   \havecopyrightspacefalse}

% 'widereview' option.

\DeclareOption{widereview}{%
   \vgtc@camerafalse%
   \vgtc@preprintfalse%
   \vgtc@reviewtrue%
   \renewcommand{\vgtc@columnmode}{onecolumn}%
   \vgtc@widetrue%
   \vgtc@doublespacedtrue%
   \renewcommand{\vgtc@pointsize}{12pt}}

% 'preprint' option.

\DeclareOption{preprint}{%
   \vgtc@camerafalse%
   \vgtc@preprinttrue%
   \vgtc@reviewfalse%
   \renewcommand{\vgtc@columnmode}{twocolumn}%
   \vgtc@doublespacedfalse%
   \renewcommand{\vgtc@pointsize}{9pt}}

\DeclareOption{tvcgpapersize}
  {\setlength\paperheight {10.75in}%
   \setlength\paperwidth  {7.875in}
   \tvcgsizetrue
}

% 'journal' option
\DeclareOption{journal}{%
   \vgtcjournaltrue %
   \@twosidetrue  \@mparswitchtrue %
   %\ExecuteOptions{tvcgpapersize}
}


%%% the following code has partly been adapted from egpubl.cls
%\newif\ifpdf  % determine if we are running PDFLaTeX or not
%\ifx\pdfoutput\undefined \pdffalse % we are not running PDFLaTeX 
%\else 
% %\pdfoutput=1 % we are running PDFLaTeX 
% \pdftrue 
%\fi

\newcommand{\vgtcinsertpkg}{}
\DeclareOption{electronic}{%
 \renewcommand{\vgtcinsertpkg}{%
  \RequirePackage[pagebackref,bookmarks]{hyperref}
  \RequirePackage[all]{hypcap}% makes hyperlinks go to top of floats rather than the caption
  \RequirePackage[svgnames]{xcolor}
  
  \hypersetup{
   pdfpagemode=UseNone,
   pdftitle={\@title},
   pdfauthor={\@author},
   pdfsubject={VGTC Conference Paper},
   pdfkeywords={\vgtc@keywords},
   pageanchor=true,
   plainpages=false,       % for problems with page referencing
   hypertexnames=false,    % for handling subfigures correctly
   bookmarksnumbered=false,% include the section numbers in the list
   bookmarksopen=true,     % In the list, display highest level only
   bookmarksopenlevel=3,   % display three levels of bookmarks
   pdfpagemode=UseNone,    % show just the page
   pdfstartview=Fit,       % defail page view is the whole page at once
   pdfborder={0 0 0},      % we don't want those silly boxes for links
   breaklinks=true,        % allow breaking links
   colorlinks=true,        % color all links blue
   linkcolor=NavyBlue,
   urlcolor=NavyBlue,
   citecolor=NavyBlue,
   anchorcolor=black,      % changing this colors te teaser caption using dvips
   nesting=true,
   linktocpage,
   pdfdisplaydoctitle}
   \ifpdf%                    % if we use pdflatex
   \else%                     % else we use pure latex
    \renewcommand{\pdfbookmark}[3][]{}
   \fi

     %% it is recommended to use ``\cref{sec:bla}`` instead of ``Sec.~\ref{sec:bla}''
  %% the following lines load cleveref for consistent, style-dependent cross-references
  %% then customize how sections and tables are referenced. Note that both ``\crefname``
  %% AND ``\Crefname`` must be specified for each, otherwise cleveref uses whatever is
  %% given for both cases.
  \RequirePackage[nameinlink,capitalise]{cleveref}
  \crefname{table}{Tab.}{Tabs.}
  \Crefname{table}{Table}{Tables}
  \crefname{section}{Sec.}{Secs.}
  \Crefname{section}{Section}{Sections}
  
 }
}
%% end of code adapted from egpubl.cls

% Assume, for the moment, that we're preparing a 'cameraready' version.

\ExecuteOptions{electronic,cameraready}
%\AtBeginDocument{\vgtcinsertpkg}

% Here's a warning command for use just below.

\newcommand{\vgtc@optwarning}[1]{%
   \ifvgtc@camera
       \ClassWarningNoLine{vgtc}%
           {#1 option used in camera-ready mode.\MessageBreak
            This violates submission specifications}
   \fi
}

% The 'onecolumn' option doesn't work in 'cameraready' mode.

\DeclareOption{onecolumn}{%
   \vgtc@optwarning{onecolumn}
   \renewcommand{\vgtc@columnmode}{onecolumn}}

% The 'twocolumn' option works in 'cameraready' mode.

\DeclareOption{twocolumn}{%
   \renewcommand{\vgtc@columnmode}{twocolumn}}

% Only the '9pt' size works in 'cameraready' mode.

\DeclareOption{9pt}{%
   \renewcommand{\vgtc@pointsize}{9pt}}
\DeclareOption{10pt}{%
   \vgtc@optwarning{10pt}
   \renewcommand{\vgtc@pointsize}{10pt}}
\DeclareOption{11pt}{%
   \vgtc@optwarning{11pt}
   \renewcommand{\vgtc@pointsize}{11pt}}
\DeclareOption{12pt}{%
   \vgtc@optwarning{12pt}
   \renewcommand{\vgtc@pointsize}{12pt}}

% The 'singlespace' option works in 'cameraready' mode.

\DeclareOption{singlespace}{%
   \vgtc@doublespacedfalse}

% The 'doublespace' option does not work in 'cameraready' mode.

\DeclareOption{doublespace}{%
   \vgtc@optwarning{doublespace}
   \vgtc@doublespacedtrue}

% No 'titlepage' option in 'cameraready' mode.

\DeclareOption{titlepage}{%
   \OptionNotUsed%
   \ClassWarningNoLine{vgtc}{titlepage option not allowed}}

% No 'landscape' mode in 'cameraready' mode, either.

\DeclareOption{landscape}{%
   \OptionNotUsed%
   \ClassWarningNoLine{vgtc}{landscape option not allowed}}

% Pass everything else to the 'article' class, upon which this is based.

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\PassOptionsToClass{\vgtc@columnmode}{article}
\ifvgtcjournal
\PassOptionsToClass{twoside}{article}
\fi
\ifdim\vgtc@pointsize>9pt
   \PassOptionsToClass{\vgtc@pointsize}{article}
\fi

% ------------ package loading --------------

\LoadClass{article}

% ------------ main code --------------
\newcommand{\vgtc@onlineid}{}
\newcommand{\onlineid}[1]{\renewcommand{\vgtc@onlineid}{#1}}

%\newcommand{\vgtc@preprinttext}{To appear in IEEE Transactions on Visualization and Computer Graphics}
%\newcommand{\preprinttext}[1]{\renewcommand{\vgtc@preprinttext}{#1}}

\newcommand{\vgtc@DOI}{xx.xxxx/TVCG.201x.xxxxxxx/}

\newcommand{\vgtc@preprinttext}{\parbox{0.9\textwidth}{$\copyright$ \the\year~IEEE. This is the author's version of the article that has been published in the proceedings of IEEE Visualization conference. The final version of this record is available at: \href{https://doi.org/\vgtc@DOI}{\color{blue}\vgtc@DOI}}}



\newcommand{\preprinttext}[1]{\renewcommand{\vgtc@preprinttext}{#1}}
\newcommand{\ieeedoi}[1]{\renewcommand{\vgtc@DOI}{#1}}


\newcommand{\vgtc@reviewtext}{Online Submission ID: \vgtc@onlineid}
\newcommand{\reviewtext}[1]{\renewcommand{\vgtc@reviewtext}{#1}}
\newcommand{\shortauthortitletext}{}
\newcommand{\shortauthortitle}[1]{\renewcommand{\shortauthortitletext}{#1}}
%\newcommand{\journalidtxt}{IEEE TRANSACTIONS ON VISUALIZATION AND
%COMPUTER GRAPHICS, VOL. 12, NO. 5, SEPTEMBER/OCTOBER 2006}
%\newcommand{\journalid}[1]{\renewcommand{\journalidtxt}{#1}}

%%Comment from here
\newcommand{\manuscriptnotetxt}{
Manuscript received xx xxx. 201x; accepted xx xxx. 201x. Date of Publication xx xxx. 201x; date of current version xx xxx. 201x. For information on obtaining reprints of this article, please send  e-mail to: reprints@ieee.org. Digital Object Identifier: xx.xxxx/TVCG.201x.xxxxxxx
}
%% to here to not use the blank

%% NOTE FOR BLANK SPACING
%% uncomment from here
%\newcommand{\manuscriptnotetxt}{\vspace{.25in}}
%%  to here
\newcommand{\manuscriptnote}[1]{\renewcommand{\manuscriptnotetxt}{#1}}
\newcommand{\copyrighttext}{}
%%\newcommand{\copyrighttext}{1077-2626/12/\$31.00 \copyright 2014 IEEE%
%%\hskip0.5in Published by the IEEE Computer Society}
\newcommand{\authorfootertext}{}
\newcommand{\authorfooter}[1]{\renewcommand{\authorfootertext}{{\em#1}}}
\newcommand{\firstsectiontxt}{}
\newcommand{\firstsection}[1]{\renewcommand{\firstsectiontxt}{#1}}
\newcommand{\acknowledgments}[1]{%
\ifvgtc@review\else%
\section*{Acknowledgments}
#1
\fi}

\newcommand{\ifcamera}[1]{\ifvgtc@camera #1 \fi}
\newcommand{\ifreview}[1]{\ifvgtc@review #1 \fi}
\newcommand{\ifcameraelse}[2]{\ifvgtc@camera #1 \else #2 \fi}
\newcommand{\ifreviewelse}[2]{\ifvgtc@review #1 \else #2 \fi}

\ifvgtcjournal
\renewcommand{\figurename}{Fig.}

\setlength{\textheight}{9.625in}
\setlength{\topmargin}{-0.625in}
\setlength{\headheight}{0.0625in}
\setlength{\headsep}{0.250in}
\setlength{\footskip}{0.25in}
\flushbottom
\setlength{\textwidth}{7.125in}
\setlength{\columnsep}{0.17in}
\newlength{\captionmargin}
\setlength{\captionmargin}{0in}

\iftvcgsize
 \setlength\paperheight {10.75in}
 \setlength\paperwidth  {7.875in}
 \setlength{\evensidemargin}{-0.6875in} %1in-0.3125
 \setlength{\oddsidemargin}{-.58in} %1in-0.4375
\else %else assume letter
 \setlength\paperheight {11in}
 \setlength\paperwidth  {8.5in}
 \setlength{\evensidemargin}{-0.375in} %1-(0.3125+0.3125)
 \setlength{\oddsidemargin}{-.25in} %1-(0.4375+0.3125)
\fi


\else
% conference template margins
\setlength{\textheight}{9.25in}
\setlength{\topmargin}{-0.700in}
\setlength{\headheight}{0.2in}
\setlength{\headsep}{0.250in}
\setlength{\footskip}{0.5in}
\flushbottom
\setlength{\textwidth}{7in}
\setlength{\oddsidemargin}{-0.25in}
\setlength{\evensidemargin}{-0.25in}
\setlength{\columnsep}{2pc}
%%\setlength{\parindent}{1em}
\newlength{\captionmargin}
\setlength{\captionmargin}{0in}
\fi

% adjust marginpars so that they can be used for comments during editing
\ifvgtcjournal % for journal style
\addtolength{\marginparwidth}{-1.5mm} % fix margin
\else % conference style
\addtolength{\marginparwidth}{16mm}
\fi
\addtolength{\marginparsep}{-2mm} % fix margin

\renewcommand{\ps@plain}%
   {%
   \renewcommand{\@oddhead}{}%
   \renewcommand{\@oddfoot}{}%

   \ifvgtc@preprint%
     \renewcommand{\@oddhead}{\hfil\textit{\vgtc@preprinttext}\hfil}%
     \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
   \fi%

   \ifvgtc@review%
     \renewcommand{\@oddhead}{\hfil\textit{\large\vgtc@reviewtext}\hfil}%
     \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
   \fi%

   \ifvgtcjournal%
     \ifvgtc@review\else\ifvgtc@preprint\else%
%        \renewcommand{\@evenhead}{\hfil\sffamily\small\MakeUppercase{\journalidtxt}}%
%        \renewcommand{\@oddhead}{\sffamily\small\MakeUppercase{\shortauthortitletext}\hfil}%
       \renewcommand{\@oddfoot}{}% no page number
       \renewcommand{\@evenfoot}{\@oddfoot}%
     \fi\fi%
   \else%

     \renewcommand{\@evenhead}{\@oddhead}%
     \renewcommand{\@evenfoot}{\@oddfoot}%
   \fi%
   }

% will be used on the first page
\renewcommand{\ps@empty}%
   {%
   \renewcommand{\@oddhead}{}%
   \renewcommand{\@oddfoot}{}%

   \ifvgtc@preprint%
     \renewcommand{\@oddhead}{\hfil\textit{\large\vgtc@preprinttext}\hfil}%
     \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
   \fi%

   \ifvgtc@review%
     \renewcommand{\@oddhead}{\hfil\textit{\large\vgtc@reviewtext}\hfil}%
     \renewcommand{\@oddfoot}{\hfil\textrm{\thepage}\hfil}%
   \fi%

   \ifvgtcjournal%
     \ifvgtc@review\else\ifvgtc@preprint\else%
%        \renewcommand{\@oddhead}{\sffamily\small\MakeUppercase{\journalidtxt}\hfil}%
       \renewcommand{\@oddfoot}{\hfil{\small\sffamily\copyrighttext}\hfil}%
     \fi\fi%
   \fi%
   \renewcommand{\@evenhead}{\@oddhead}%
   \renewcommand{\@evenfoot}{\@oddfoot}%
   }

% no page numbers - they are added in production

\pagestyle{plain}

\newcommand{\vgtc@setninepoint}{
   \renewcommand\normalsize{%
       \@setfontsize\normalsize\@ixpt\@xpt
       \abovedisplayskip 9\p@ \@plus2\p@ \@minus4\p@
       \abovedisplayshortskip \z@ \@plus3\p@
       \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
       \belowdisplayskip \abovedisplayskip
       \let\@listi\@listI}
   \renewcommand\small{%
       \@setfontsize\small\@viipt\@ixpt
       \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
       \abovedisplayshortskip \z@ \@plus2\p@
       \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
       \def\@listi{\leftmargin\leftmargini
                   \topsep 4\p@ \@plus2\p@ \@minus2\p@
                   \parsep 2\p@ \@plus\p@ \@minus\p@
                   \itemsep \parsep}%
       \belowdisplayskip \abovedisplayskip}
   \renewcommand\footnotesize{%
       \@setfontsize\footnotesize\@viiipt{9.5}%
       \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
       \abovedisplayshortskip \z@ \@plus\p@
       \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
       \def\@listi{\leftmargin\leftmargini
                   \topsep 3\p@ \@plus\p@ \@minus\p@
                   \parsep 2\p@ \@plus\p@ \@minus\p@
                   \itemsep \parsep}%
       \belowdisplayskip \abovedisplayskip}
   \renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
   \renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
   \renewcommand\large{\@setfontsize\large\@xpt\@xiipt}
   \renewcommand\Large{\@setfontsize\Large\@xiipt{14}}
   \renewcommand\LARGE{\@setfontsize\LARGE\@xivpt{18}}
   \renewcommand\huge{\@setfontsize\huge\@xviipt{22}}
   \renewcommand\Huge{\@setfontsize\Huge\@xxpt{25}}
   \selectfont
}
\ifdim\vgtc@pointsize=9pt
   \vgtc@setninepoint
\fi

\newcommand{\vgtc@sectionfont}{}
\newcommand{\sectionfont}[1]{\renewcommand{\vgtc@sectionfont}{#1}}
\renewcommand\section{\@startsection{section}{1}{\z@}%
               {-2ex \@plus -1ex \@minus -.2ex}%
               {0.8ex \@plus .2ex}%
               {\reset@font\normalsize\sffamily\bfseries\scshape\vgtc@sectionfont}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
               {-1.8ex\@plus -1ex \@minus -.2ex}%
               {0.8ex \@plus .2ex}%
               {\reset@font\normalsize\sffamily\bfseries\vgtc@sectionfont}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
               {-1.8ex\@plus -1ex \@minus -.2ex}%
               {0.8ex \@plus .2ex}%
               {\reset@font\sffamily\normalsize\vgtc@sectionfont}}

\renewcommand\paragraph{\@startsection{paragraph}{4}{1em}%
               {1ex \@plus 1ex \@minus.2ex}%
               {-1em}%
               {\reset@font\normalsize\sffamily\vgtc@sectionfont}}
%\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
%                {3.25ex \@plus1ex \@minus.2ex}%
%                {-1em}%
%                {\reset@font\normalsize\sffamily\bfseries\vgtc@sectionfont}}

\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
               {3.25ex \@plus1ex \@minus .2ex}%
               {-1em}%
               {\reset@font\normalsize\sffamily\bfseries\vgtc@sectionfont}}

\ifvgtc@wide\else
%% make captionfont 8pt
 \newcommand{\captionfonts}{\scriptsize\sffamily}

 \def\instring#1#2{TT\fi\begingroup
   \edef\x{\endgroup\noexpand\in@{#1}{#2}}\x\ifin@}
 \long\def\@makecaption#1#2{%
   \if\instring{Table}{#1}\else\vskip\abovecaptionskip\fi
   \leftskip = \captionmargin \rightskip = \leftskip%
   \sbox\@tempboxa{\captionfonts #1\ifvgtcjournal. \else: \fi #2}%
   \ifdim \wd\@tempboxa >\hsize
     {\captionfonts #1\ifvgtcjournal. \else: \fi #2\par}
   \else %single line caption
     \global \@minipagefalse
     \def\@figcaptype{figure}
     \hskip0.0in%needed to make leftskip work
     \hb@xt@\hsize{\ifvgtcjournal\ifx\@captype\@figcaptype\else\hfil\fi\else\hfil\fi\box\@tempboxa\hfil}%
   \fi
   \if\instring{Table}{#1}\vskip.5\abovecaptionskip\else\vskip\belowcaptionskip\fi}

%% fix the font size of the bibliography to 8pt
 \newdimen\bibindent
 \newdimen\bibspacing
 \setlength\bibindent{1em}
 \setlength{\bibspacing}{\z@}
 \renewenvironment{thebibliography}[1]
      {\section*{\refname}%
       \scriptsize%
       \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
       \list{\@biblabel{\@arabic\c@enumiv}}%
            {\settowidth\labelwidth{\@biblabel{#1}}%
             \leftmargin\labelwidth
             \advance\leftmargin\labelsep
             \itemsep\bibspacing    % should this be commented out?
             \parsep\bibspacing     % should this be commented out?
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
\fi

\newcommand{\vgtc@empty}{}

\newcommand{\vgtc@affiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\vgtc@affiliation}{#1}}

\newcommand{\vgtc@category}{}
\newcommand{\category}[1]{\renewcommand{\vgtc@category}{#1}}
\newcommand{\vgtccategory}[1]{\category{#1}}
\vgtccategory{n/a}
\newcommand{\vgtc@papertype}{}
\newcommand{\papertype}[1]{\renewcommand{\vgtc@papertype}{#1}}
\newcommand{\vgtcpapertype}[1]{\papertype{#1}}

%\newcommand{\vgtc@format}{}
%\newcommand{\format}[1]{\renewcommand{\vgtc@format}{#1}}
%\newcommand{\vgtcformat}[1]{\format{#1}}
\newcommand{\vgtcformat}[1]
{\@latex@warning{Format specification no longer required.}}

\RequirePackage{etoolbox}

\newcommand{\vgtc@teaser}{}
  \newcommand{\teaser}[1]{%
    \renewcommand{\vgtc@teaser}{%
    %\setlength{\captionmargin}{0.33in}% TI: no longer needed since the entire teaser is now limited to the width of the abstract
	  \newlength{\abstractwidth}%
	  \setlength{\abstractwidth}{\linewidth}%
    \addtolength{\abstractwidth}{-.66in}%
	  \begin{minipage}{\abstractwidth}
    #1%
		\end{minipage}
	  \setlength{\captionmargin}{0in}%
	  }%
	}

\newcommand{\vgtc@abstxt}{}
\let\origabstract\abstract
\let\endorigabstract\endabstract
\renewcommand{\abstract}[1]{\renewcommand{\vgtc@abstxt}{#1}}

\newcommand{\vgtc@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\vgtc@keywords}{#1}}

\newcommand{\vgtc@indexterms}{}
\newcommand{\CCScatlist}[1]{\renewcommand{\vgtc@indexterms}{#1}}

\newcommand{\CCScat}[4]{%
    #1 [#2]%
    \ifx#3\vgtc@empty \else : #3\fi%
    \ifx#4\vgtc@empty \else ---#4\fi%
}

\newcommand{\CCScatTwelve}[4]{%
    #1%
		\ifx#2\vgtc@empty \else ---#2\fi%%
		\ifx#3\vgtc@empty \else ---#3\fi%
    \ifx#4\vgtc@empty \else ---#4\fi%
}



% use any of the following to adjust spaces in title block
\newlength{\titlespace}
\setlength{\titlespace}{0.25in}

\newlength{\teaserspace}
\setlength{\teaserspace}{0.0in}

\newlength{\abstxtspace}
\setlength{\abstxtspace}{0.20in}

\renewcommand{\@maketitle}{%
   \ifvgtc@review
       \begin{center}%
           \renewcommand{\thanks}[1]{}
           {\sffamily\ifvgtcjournal\huge\else\LARGE\bfseries\fi%
	      \vgtc@sectionfont%
	      \@title \par}%
           \vspace{1\baselineskip}%
           {Category: \vgtc@category \par}%
					 \ifx\vgtc@papertype\vgtc@empty \else%
					 {\vspace{1ex}Paper Type: \vgtc@papertype \par}%
					 \fi%
%            \vspace{0.25\baselineskip}%	% no longer needed (1996)
%            {Format: \vgtc@format \par}%	% no longer needed (1996)
           \vspace{\titlespace}%
           \ifx\vgtc@teaser\vgtc@empty \else%
               \begingroup%
                   \def\@captype{figure}%
                   \vgtc@teaser%
               \endgroup\par%
               \vspace{\teaserspace}%
           \fi%
       \end{center} \par%
   \else
       \begin{center}%
           {\sffamily\ifvgtcjournal\huge\else\LARGE\bfseries\fi%
	      \vgtc@sectionfont%
	      \@title \par}%
           \ifvgtcjournal%
             %\vspace{2\baselineskip}%
             \vspace{14pt}%
           \else%
	    \vspace{1\baselineskip}\fi%
           \large\sffamily\vgtc@sectionfont 
           \begin{tabular}[t]{c}%
               \@author
           \end{tabular}\par%
           \ifx\vgtc@affiliation\vgtc@empty \else%
               \par\vspace{1\baselineskip}%
               \vgtc@affiliation\par%
           \fi%
          \ifvgtcjournal\vspace{0.08in}\else\vspace{\titlespace}\fi%
           \ifx\vgtc@teaser\vgtc@empty \else%
             { 
               \begingroup%
                   \def\@captype{figure}%
                   \vgtc@teaser%
               \endgroup\par%
             }
               \vspace{\teaserspace}%
           \fi%
       \end{center} \par%
   \fi
   \ifvgtcjournal%
       {\scriptsize\sffamily%\renewcommand{\baselinestretch}{1.1}
       \leftskip = 0.33in \rightskip = \leftskip%
       \ifx\vgtc@abstxt\vgtc@empty \else%
            \begingroup%
                {\bfseries Abstract}---\vgtc@abstxt%
            \endgroup\par%
        \fi%
        \ifx\vgtc@keywords\vgtc@empty \else%
            \begingroup%
                %{\normalsize\vgtc@absfont {\bfseries Keywords - } \vgtc@keywords}%
                \vspace{0.5\baselineskip}%
                \par\noindent \textbf{Index Terms}---\vgtc@keywords%
            \endgroup\par%
        \fi%
%%          \ifx\vgtc@indexterms\vgtc@empty \else%
%%              \begingroup%
%%      %            {\normalsize\vgtc@absfont {\bfseries Index Terms - } %
%%      %             \vgtc@indexterms}%
%%                  \vspace{0.5\baselineskip}%
%%                  \par\noindent \textbf{Index Terms -} \vgtc@indexterms%
%%              \endgroup\par%
%%          \fi%
        }%
	\begin{center}\includegraphics{diamondrule}\end{center}
       %\vspace{\abstxtspace}%
	\ifx\firstsectiontxt\vgtc@empty \else
	  \section{\firstsectiontxt}
	\fi
    \fi%
}

\let\vgtc@origmaketitle\maketitle
\let\vgtc@origand\and
\renewcommand{\maketitle}{%
   \let\vgtc@title\@title%
   \let\vgtc@author\@author%
   \vgtc@origmaketitle%
   \thispagestyle{empty}%
   \ifvgtc@doublespaced%
       \renewcommand{\baselinestretch}{1.66}\selectfont%
   \fi%
   \ifvgtcjournal% no copyrightspace for journal, but authorfooter
     \ifvgtc@review\else
       \renewcommand{\thefootnote}{}%
       \footnotetext[0]{
         \begin{flushleft}
         \vskip -6pt
         \begin{list}{\textbullet}{
           \setlength{\partopsep}{0pt}
           \setlength{\topsep}{0pt}
           \setlength{\itemsep}{-2pt}
           \setlength{\itemindent}{-4pt}
           \setlength{\leftmargin}{12pt}}
           \authorfootertext
         \end{list}
         \vskip 4pt
         \ifvgtc@preprint\else\textit{\manuscriptnotetxt}\fi
         \end{flushleft}
       }%
       \renewcommand{\thefootnote}{\arabic{footnote}}
     \fi
   \else%
       \ifhavecopyrightspace\copyrightspace\fi%
       \ifx\vgtc@abstxt\vgtc@empty \else%
            \begingroup%
                \begin{origabstract} \vgtc@abstxt \end{origabstract} %
            \endgroup\par%
            %\vspace{\abstxtspace}%
        \fi%
        \ifx\vgtc@keywords\vgtc@empty \else%
            \begingroup%
                \vspace{0.5\baselineskip}%
                \par\noindent \textbf{Index Terms: } \vgtc@keywords%
            \endgroup\par%
            %\vspace{\abstxtspace}%
        \fi%
        \ifx\vgtc@indexterms\vgtc@empty \else%
            \begingroup%
    %            {\normalsize\vgtc@absfont {\bfseries Index Terms - } %
    %             \vgtc@indexterms}%
                \vspace{0.5\baselineskip}%
                \par\noindent \textbf{Index Terms:} \vgtc@indexterms%
            \endgroup\par%
            %\vspace{\abstxtspace}%
        \fi%
    \fi%
    \ifx\firstsectiontxt\vgtc@empty \else
      \ifvgtcjournal
        \section*{}
        \vskip -1.5em
      \else
        \section{\firstsectiontxt}
      \fi
    \fi
}

%% \newtoks\vgtc@abs
%% \ifvgtc@review
%%     \long\def\vgtc@add#1{\global\vgtc@abs\expandafter{\the\vgtc@abs#1}}
%%     \long\def\vgtc@collect{%
%%         \global\vgtc@abs{}%
%%         \let\abstract\vgtc@@collect
%%         \abstract
%%     }
%%     \long\def\vgtc@@collect#1\end#2{%
%%         \def\@tempa{#2}%
%%         \ifx\@tempa\@currenvir
%%             \vgtc@add{#1}%
%%             \edef\abstract{\noexpand\end{\@tempa}}%
%%         \else
%%             \vgtc@add{#1\end{#2}}%
%%         \fi
%%         \abstract
%%     }
%%     \newcommand{\vgtc@modify}{%
%%         \let\vgtc@origabs\abstract%
%%         \let\vgtc@origendabs\endabstract%
%%         \renewenvironment{abstract}%
%%             {\vgtc@collect}%
%%             {\begingroup
%%                 \let\abstract\vgtc@origabs
%%                 \let\endabstract\vgtc@origendabs
%%                 \begin{abstract} \the\vgtc@abs \end{abstract}
%%             \endgroup}%
%%     }
%%     \AtBeginDocument{\vgtc@modify}
%% \fi

\newcommand{\keywordsprefix}{%
  \vspace{0.5\baselineskip}%q
  \par\noindent \textbf{Index Terms - } \vgtc@keywords%
}
\newenvironment{CRcatlistprefix}{%
   \vspace{0.5\baselineskip}%
   \par\noindent \textbf{CR Categories: }%
}{}

% leave a 0.5 inch space at the bottom of the left column
% on the first page for the copyright block.

\newlength{\vgtc@copyrightlength}
\setlength{\vgtc@copyrightlength}{0.5in}

\newcommand{\copyrightspace}{%
   \renewcommand{\thefootnote}{}%
   \footnotetext[0]{\rule[\vgtc@copyrightlength]{2.71828in}{0in}}%
   \renewcommand{\thefootnote}{\arabic{footnote}}
}

\newcommand{\nocopyrightspace}{\havecopyrightspacefalse}
\renewcommand{\footnoterule}{%
   \kern-3pt
   \leftline{\hskip1in\vbox{\hrule width 0.45\columnwidth}\hfil}
   \ifvgtcjournal \else \kern 2.6pt \fi
}

\newcommand{\vgtc@contactname}{}
\newcommand{\contactname}[1]{\renewcommand{\vgtc@contactname}{#1}}

\newcommand{\vgtc@contactaddress}{}
\newcommand{\contactaddress}[1]{\renewcommand{\vgtc@contactaddress}{#1}}

\newcommand{\vgtc@contactphone}{}
\newcommand{\contactphone}[1]{\renewcommand{\vgtc@contactphone}{#1}}

\newcommand{\vgtc@contactfax}{}
\newcommand{\contactfax}[1]{\renewcommand{\vgtc@contactfax}{#1}}

\newcommand{\vgtc@contactemail}{}
\newcommand{\contactemail}[1]{\renewcommand{\vgtc@contactemail}{#1}}

\newcommand{\vgtc@estpages}{}
\newcommand{\estpages}[1]{\renewcommand{\vgtc@estpages}{#1}}

\newif\ifvgtc@cover
\vgtc@coverfalse
%\ifvgtc@review
%    \vgtc@covertrue
%\else
%    \vgtc@coverfalse
%\fi

\newcommand{\suppresscover}{\vgtc@coverfalse}
\newcommand{\vgtc@coverpage}{%
   \begin{titlepage}%
       \def\thanks##1{}
       \let\and\vgtc@origand
       \vgtc@setninepoint\normalsize
       \renewcommand{\baselinestretch}{1}\selectfont
       \begin{center}%
           \vspace*{\fill}
           {\LARGE\sffamily\bfseries\vgtc@sectionfont \vgtc@title \par}%
           \vspace{2\baselineskip}%
           {\large
               \begin{tabular}[t]{c}%
               \vgtc@author
            \end{tabular}\par%
           }%
           \vspace{1\baselineskip}%
           {\large \vgtc@affiliation \par}%
           \addvspace{3\baselineskip}%
           {Category: \vgtc@category \par}%
%            \vspace{0.5\baselineskip}%		% no longer needed (1996)
%            {Format: \vgtc@format \par}%	% no longer needed (1996)
           \vspace{3\baselineskip}%
           \begin{tabular}{ll}
               Contact: & \vgtc@contactname \\[1\baselineskip]
                        & \begin{tabular}[b]{@{}l@{}}
                              \vgtc@contactaddress
                              \end{tabular} \\[1\baselineskip]
               phone: & \vgtc@contactphone \\
               fax: & \vgtc@contactfax \\
               email: & \vgtc@contactemail
           \end{tabular}\par%
           \vspace*{3\baselineskip}%
           {Estimated \# of pages: \vgtc@estpages \par}%
           \vspace*{\baselineskip}%
           {Index Terms: \vgtc@keywords \par}%
           \vspace*{\fill}%
           \begin{minipage}{5in}%
               \the\vgtc@abs
           \end{minipage}\par%
           \vspace*{\fill}
       \end{center}%
   \end{titlepage}%
}

\AtEndDocument{\ifvgtc@cover \vgtc@coverpage \fi}
\newcommand{\vgtcopening}[1]{%
 \ClassError{vgtc}{%
   The vgtcopening command is no longer needed.\MessageBreak%
   Switch to the maketitle command and abstract environment}{}}
\InputIfFileExists{vgtc.cfg}
       {\typeout{***************************************^^J%
                 * Local config file vgtc.cfg used *^^J%
                 ***************************************}}
       {}
\endinput

% End of file 'vgtc.cls'.

