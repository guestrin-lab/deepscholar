%% This is file `ios1.bst' (created: 2011-12-27)
%% Compatible with bibtex version 0.99d.
%%
%% Please submit bugs or your comments to latex-support@vtex.lt
%% Start subject with a "[BMC] bibtex: " + subject
%%
%% Sorting entries is done by name and year fields!
%%
%% You are free to use this style file as you see fit, provided 
%% that you do not make changes to the file. 
%% If you DO make changes, you are required to rename this file.
%%
%% It may be distributed under the terms of the LaTeX Project Public
%% License, as described in lppl.txt in the base LaTeX distribution.
%% Either version 1.0 or, at your option, any later version.
%%
%% LIMITATIONS:
%% If you are getting error like
%% "Sorry---you've exceeded BibTeX's number of string global-variables"
%% that means you are using to old bibtex version. You should download latest version 0.99d. 
%% (One of the binaries location is minimals.contextgarden.net/current/bin/common/)
%%
%% BibTeX programming: Linas Stonys, VTeX, Lithuania, lstonys@vtex.lt
%
% Changes:
%
% Settings:
% It's possible to pass few options to bibtex style via @settings reference. 
% To validate options, need to write that reference (@settings) citeing key (f.e: \citation{foo}) 
% to aux file. From tex file it can be done with \nocite{} command.
%
% Examples:
%   1) To make unsorted bibliography (bibs listed as cited)  
%     a) write to bib file:
%       @settings{label, options="unsort"}
%     b) write in tex file \nocite{label}
%   
%   2) To make author-year bibliography
%     a) write to bib file:
%       @settings{label, options="nameyear"}
%     b) write in tex file \nocite{label}
%   
%   3) To make alphanumerical unsorted bibliography (bibs listed as cited)  
%     a) write to bib file:
%       @settings{label, options="alpha,unsort"}
%     b) write in tex file \nocite{label}
% 
% By default, style produces unsorted bibliography. 
%

ENTRY
  { address
    author
    booktitle
    chapter
    doi
    edition
    editor
    howpublished
    institution
    isbn
    issn
    journal
    key
    month
    note
    number
    organization
    pages   
    publisher
    school
    series
    title
    type
    volume
    year
    url
    info
    options
  }
  {}
   { label extra.label sort.label short.list }

INTEGERS { output.state before.all mid.sentence after.sentence
           after.block after.authors between.elements bother slen set.settings default.info
           organization.outputed done}

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
  #4 'after.authors :=
  #5 'between.elements :=
}

STRINGS { s t  element}
STRINGS { longest.label last.label list.string}
STRINGS { v l f j b temp.str}

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { " " * write$ newline$}
    { output.state after.block =
        { add.period$ write$
          newline$
        }
        {
          output.state after.authors =
          { ", " * write$
            newline$
          }
          { output.state between.elements =
              { ", " * write$ newline$}
              { output.state before.all =
                   'write$
                  { add.period$ " " * write$ newline$}
                  if$
              }
              if$
          }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}
FUNCTION {fin.entry}
{
  "." *
  write$ newline$
}

FUNCTION {colon}
{ after.authors 'output.state := }


FUNCTION {insert.comma}
{ output.state before.all =
    'skip$
    { between.elements 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}
INTEGERS {sk}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {sort.format.month}
{ 't :=
      t #1 #2 substring$ "01" =
      t #1 #1 substring$ "1" =
      t #2 #2 substring$ "" =
      and or
      { "January" }
    { t #1 #2 substring$ "02" =
            t #1 #1 substring$ "2" =
            t #2 #2 substring$ "" =
            and or
      { "February" }
      { t #1 #2 substring$ "03" =
             t #1 #1 substring$ "3" =
             t  #2 #2 substring$ "" =
             and or
        { "March" }
        { t #1 #2 substring$ "04" =
            t #1 #1 substring$ "4" =
            or
          { "April" }
          { t #1 #2 substring$ "05" =
               t #1 #1 substring$ "5" =
               or
            { "May" }
            { t #1 #2 substring$ "06" =
                t #1 #1 substring$ "6" =
                or
              { "June" }
              { t #1 #2 substring$ "07" =
                  t #1 #1 substring$ "7" =
                  or
                { "July" }
                { t #1 #2 substring$ "08" =
                       t #1 #1 substring$ "8" =
                       or
                  { "August" }
                  { t #1 #2 substring$ "09" =
                    t #1 #1 substring$ "9" =
                    or
                    { "September" }
                    { t #1 #2 substring$ "10" =
                      { "October" }
                      { t #1 #2 substring$ "11" =
                        { "November" }
                        { t #1 #2 substring$ "12" =
                          { "December" }
                          { t } % No match
                        if$
                        }
                      if$
                      }
                    if$
                    }
                  if$
                  }
                if$
                }
              if$
              }
            if$
            }
          if$
          }
        if$
        }
      if$
      }
    if$
    }
  if$
 
}

FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}


FUNCTION {make.tag}
{ 't :=
  duplicate$ empty$
    { pop$ "" }
    { bother #0 =
        { 
            t "btitle" =
            t "jtitle" =
            or
                {"\textit{"  swap$ * "}" * }
                'skip$
            if$
            t "volume" =
                {"\textbf{"  swap$ * "}" * }
                'skip$
            if$
        }
        { "\b" t * "{" * swap$ * "}" * }
      if$
    }
  if$
}

INTEGERS { multiresult char.num k int}
INTEGERS {str.length count return save.num1 save.num2}
FUNCTION {is.in.list}{
    'f :=
    %%%% Saving original "s" and "t"
    s 'j :=
    f 's :=
    t 'f :=  
  list.string text.length$ 'str.length :=
  list.string 't :=
  int   'save.num1 :=
  count 'save.num2 :=
  #1 'int :=
  #1 'count :=
  #0 'return :=
    
    { count str.length = not 
      #0 str.length = not
      and
    }
    {
            count #1 + 'count :=
          t int #1 substring$ "," =
          {
            t #1 int substring$ s "," * =
              {
                     #1 'return :=
                    str.length 'count :=
                     }
                    'skip$
         if$
            
            int #1 + 'int :=
               t int global.max$ substring$ 't :=
                  #0 'int :=
          }
          'skip$
      if$ 
      int #1 + 'int :=
    }
  while$

  save.num1 'int   :=
  save.num2 'count :=
  
  #0 return =
      { s t =
            {#1 'return :=}
            'skip$
          if$
      }
      'skip$
  if$
  %%%% Returning original "s" and "t"
  j 's :=
  f 't :=  
  
  return    
}

FUNCTION {byear}
{ duplicate$ empty$
    { pop$ "" }
    {'t :=
     t text.length$ 'sk :=
            #5 sk =
                    {t #1 #4 substring$ 's :=
                     t #5 #1 substring$ 'longest.label :=
                         longest.label is.num not 
                            {s "year" make.tag  longest.label * }
                            {t "year" make.tag }
                        if$
                    }
                    {t "year" make.tag 
                          "nameyear" is.in.list           
                            {extra.label *}
                            'skip$
                        if$
                    }
            if$
    }
  if$
}

FUNCTION {format.date}
{ year empty$
    { "" }
    { year byear }
  if$
}

FUNCTION {f.date}
{ year empty$
    'skip$
    { ", "  year  byear * *}
  if$
}

function{title.lowerwords}{
"a,an,and,of,the,for,in,to,over,from,on,or,so,than,that,this,as,but,with,by,at,down,up,upon,al,ale,aus,aux,d'Analyse,d'Analystes,d'Architecture,d'Etudes,d'Histoire,da,dans,de,degli,dei,del,"
"dell',dell'Accademia,dell'Istituto,dell'Universita,della,delle,dem,der,des,detta,di,die,din,dlia,do,du,e,eXpress,ed,ego,el,em,en,et,fur,i,iz,l'Academie,l'Age,l'Antiquite," *
"l'Approximation,l'Ecole,l'Histoire,l'I.S.U.P.,l'INSA,l'IREM,l'Indirizzo,l'Institut,l'Institute,l'UFR,l'Union,l'Universite,la,las,les,na,nella,per,po,si,sue,sur,ta,tes,und,van,voor,zu,zum,zur" *
}

FUNCTION {title.case}{
    't :=
    "" 's :=
  #1 'int :=
  #1 'count :=
  list.string  'v :=
  "" 'b :=
  title.lowerwords 'list.string :=  
      
  t text.length$ 'slen :=
  
    {count slen = not }
    {t int #1 substring$ " " =
          {
            int #1 - 'int :=
            t #1 int substring$ 'b :=
          
            b is.in.list not
                {
                    b "u" change.case$ "t" change.case$ 'b :=
                }
                'skip$
            if$
            
            int #1 + 'int :=

            s b " " *  * 's :=
            int #1 + 'int :=
            t int global.max$ substring$ 't :=
            #0 'int :=
          }
          'skip$
      if$ 
      int #1 + 'int :=
      count #1 + 'count :=
    }
  while$
  v 'list.string := 
  s t "u" change.case$ "t" change.case$  * %top$    
}


FUNCTION {format.organization}
{ organization empty$
    { "" }
    { #1 'organization.outputed :=
      organization 
    }
  if$
}

FUNCTION {format.pub.address}
{ publisher empty$
    {organization empty$
        'skip$
        { author empty$ editor empty$ and
                'skip$
                {insert.comma format.organization "institute" make.tag output insert.comma}
            if$
        }
      if$ 
      address empty$
        {format.date output}
        { insert.comma
            address
            type$ "inproceedings" =
                {"conflocation"}
                {"location"}
             if$  
            make.tag 
            f.date output
        }
      if$
    }
    {
      publisher "publisher" make.tag output
      insert.comma
      address "location" make.tag  output
      insert.comma
      format.date output               
    }
  if$
}
INTEGERS { nameptr namesleft numnames tmp}
FUNCTION {mk.tag}
{ 'temp.str :=
  duplicate$ empty$
    { pop$ "" }
    { 
        b "nothing" =
            'skip$
            {"\b" temp.str * "{" * swap$ * "}" * }
        if$
    }
  if$
}

FUNCTION {space.after.dot}
{'j :=
    t 'f :=
    j 't := 
    s 'j :=
    "" 'l :=
    "" 's :=
    
  { t empty$ not }
  {   
    t #1 #3 substring$ "{-}" =
         {l  " {q}. " * 'l :=
            t #4 global.max$ substring$ 't :=
         }
            'skip$  
    if$

      t #1 #1 substring$ 's :=
          l  s * 'l := 

      s "." = 
        { 

              t #2 #1 substring$ " " =
              'skip$
               {
                 l  " " * 'l := 
               t #2 #4 substring$ "{\,}" =
                       {
                        t #5 global.max$ substring$ 't :=
                       }
                   'skip$
                   if$ 
              } 
                if$  
        } 
        'skip$
      if$

        t #2 global.max$ substring$ 't :=
    }
  while$
  f 't :=
  j 's :=
  l 
}

FUNCTION {fix.inits}{
    't :=
     "" 'element :=
      { t empty$ not }
      {  

      t #1 #3 substring$ ".q." =
         {element  ".-" * 'element :=
              t #3 global.max$ substring$ 't :=
         }
         {element  t #1 #1 substring$ * 'element :=}
         if$
        
        t #2 global.max$ substring$ 't :=
    }
  while$
  element
}

FUNCTION {fix.snm}{
    't :=
     "" 'element :=
      { t empty$ not }
      {  

      t #1 #3 substring$ "{-}" =
         {element  "-" * 'element :=
              t #3 global.max$ substring$ 't :=
         }
         {element  t #1 #1 substring$ * 'element :=}
         if$
        
        t #2 global.max$ substring$ 't :=
    }
  while$
  element
}

FUNCTION {enbrace.dash}{
    't :=
     "" 'element :=
      { t empty$ not }
      {  
      t #1 #1 substring$ "-" =
        {element  "{-}" * 'element :=}
         {element  t #1 #1 substring$ * 'element :=}
        if$
        t #2 global.max$ substring$ 't :=
    }
  while$
  element
}

FUNCTION {fix.name}{  

    enbrace.dash 's :=
    "" 'l :=
    "" 't :=

    #1 'nameptr :=
    s num.names$ 'numnames :=
    numnames 'namesleft :=
    { namesleft #0 > }
    {
      s nameptr "{vv~}{ll}{, jj}" format.name$
      s nameptr "{, ff}" format.name$ space.after.dot * 'l :=
      #1 nameptr = 
        {l 't :=}
        {t " and " * l * 't := }
      if$
      nameptr #1 + 'nameptr := 
      namesleft #1 - 'namesleft :=
    }
  while$
  t 
}

FUNCTION {name.separator}{
  namesleft #1 >
      { namesleft #2 =
         {tmp #0 = 
                {" and "}
                {" "}
              if$
              }
           {", "}
       if$
       *
    }
    'skip$
    if$
}


FUNCTION {format.author.editor}
{ 'b :=

    bother #1 =
    b "nothing" =
    or
        'skip$
        {"b" b * 'b :=}
    if$

       b "editor"  =
       b "beditor" =
       b "nothing" =
       or or
        {editor 's :=}
        {author 's :=}
    if$
    
     s fix.name 's :=
    

   "" 't := 
 
    #1 'nameptr :=
      s num.names$ 'numnames :=
      numnames 'namesleft :=
      
      s numnames "{ll}" format.name$ 'l :=
      l  "others" =
       l  "{others}" =
       l  "et al." =
       l  "{et al.}" =
       or or or
           {#1 'tmp :=}
           {#0 'tmp :=}
      if$
      
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}" format.name$ fix.snm   "snm"      mk.tag  'l :=   
      s nameptr "{f{.}.}"   format.name$ fix.inits "inits"    mk.tag  'f := 
      s nameptr "{jj}"      format.name$           "suffix"   mk.tag  'j := 
        
        
           namesleft #1 =
           tmp #1 = 
           and
           {  "\betal" 
              #1 'tmp :=
           }
           {

               b "bauthor" =
               b "beditor" =
               or
                  {"\oauthor{"}
                  {
                       b "nothing" =
                           {""}
                          {"\b" b * "{" *}
                    if$
                  }
              if$     

               f empty$
                  'skip$
                  {f * "~" *}     
              if$

              l empty$
                'skip$
                  { l *}    
              if$     

               j empty$
                'skip$
                  {" "  * j *}    
              if$

               b "nothing" =
                   {""}
                  {"}"}  
              if$
            
            * "" 't :=

          } 
          if$

          name.separator

      write$ 
          
      namesleft #1 >
        b "nothing" = not
        and
            {namesleft #2 =
             tmp #1 =
             and
                'skip$
                'newline$
             if$
            }
            'skip$
        if$

      nameptr #1 + 'nameptr := 
      namesleft #1 - 'namesleft :=
    }
  while$
  
}

FUNCTION {format.editors}
{ editor empty$
    'skip$
    { 
      author empty$
        'skip$
        {", " *  write$ newline$ }
      if$ 
      bother #0 =
          {author empty$
                   {"editor"}
                 {"nothing"}
              if$
            }
          {"editor"}
      if$  
      format.author.editor
      author empty$
        {
          editor num.names$ #1 >
            { " (eds)" }
            { " (ed.)" }
            if$
            output
        }
        { 
          editor num.names$ #1 >
            { ", eds" }
            { ", ed." }
          if$
        }
    if$ 
    }
  if$
}

FUNCTION {format.middle.editors}
{ author empty$
    'skip$
    {format.editors}
  if$
}

FUNCTION {format.authors}
{ author empty$
    'skip$
    {"author"  format.author.editor}
  if$
}

FUNCTION {author.or.editor}{
  author empty$
     { editor empty$
           'skip$
        {format.editors colon}
      if$
     }
     { format.authors colon}
   if$
}

FUNCTION {author.or.editor.or.organization}{
  author empty$ editor empty$  and 
  organization empty$ not and
    {format.organization "institute" make.tag output colon}
    'author.or.editor
  if$
}

FUNCTION {cite.author.editor}
{ 's :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ 
  #2 >
    {  " et~al." * }
    {s num.names$ #2 =
          { s #2 "{vv~}{ll}{ jj}{ f}" format.name$ "others" =
                  { " et~al."  * }
                     { " and "  s #2 "{vv~}{ll}" format.name$ * *}
                 if$
             }
             'skip$
     if$
    }
  if$
}

FUNCTION {check.auth.edit.org}
{author empty$
  {
    editor empty$
      'format.organization
      {editor cite.author.editor}
    if$   
  }
  {author cite.author.editor}
 if$  
}


FUNCTION {check.year}
{year empty$
    {""}
    {year extra.label *}
  if$ 
}



FUNCTION {output.bibitem}
{ newline$
  "\bibitem"
  "nameyear" is.in.list
   {
      "[\protect\citeauthoryear{" * write$
           check.auth.edit.org write$ 
      "}{" write$ 
           check.year write$ 
      "}]" 
    }
    'skip$
  if$
  "alpha" is.in.list
    {
      "[" * write$
      label write$
      "]" 
    }
    'skip$
  if$

  "{" * write$  
  cite$ write$
  "}" write$

  newline$
  "" before.all 'output.state :=
}

FUNCTION {string.to.integer}
{ 't :=
    t text.length$ 'k :=
  #1 'char.num :=
      { t char.num #1 substring$ 's :=
           s is.num
           s "." =
           or 
           char.num k = not
           and
              }
          {char.num #1 + 'char.num :=}
  while$
  t #1 char.num substring$  
}

FUNCTION {check.chapter}
{ #0 'char.num :=
  chapter text.length$ #10 >
    {#1 'char.num :=}
    'skip$
  if$
  char.num
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}
 
FUNCTION {find.integer}
{ 't :=
  #0 'int :=
    { int not
      t empty$ not
      and
    }
    { t #1 #1 substring$ 's :=
        s is.num 
        {#1 'int :=}
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  int 
}



FUNCTION {format.title}
{ title empty$
    { "" }
    { title }
  if$
}

FUNCTION {note.presented}{note #1 #9 substring$ "l" change.case$ "presented" = }
 
FUNCTION {n.filter}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "," =
      t #1 #1 substring$ "\" =
      t #1 #1 substring$ "~" =
      or or
            {  "" *
                  t #2 global.max$ substring$ 't :=
            }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {format.article.date}
{ year empty$
    'skip$
    { "(" year byear ")" *  * }
  if$
}


FUNCTION {format.month}{ 
  month empty$
    'skip$
    {month sort.format.month}
  if$
}

FUNCTION {formatpatent.date}
{ year empty$
    'skip$
    { month empty$
       { year }
        {format.month " " * year *}
      if$
    }
  if$
}

FUNCTION {month.year.date}
{ year empty$
    'skip$
    { month empty$
        {year byear}
        { format.month " " * year byear * }
      if$
    }
  if$
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {format.volume}
{ volume empty$
    { "book" type$ =
            {number empty$
                {""}
                {"Vol.~" number "seriesno" make.tag *}
                if$
            }
            {""} 
      if$
    }
    {"Vol.~" volume "seriesno" make.tag *}
  if$
}


FUNCTION {format.art.vol}
{ volume empty$
    'skip$
    {volume "volume" make.tag }
  if$
  number empty$
    'skip$
    { "(" number n.dashify "issue" make.tag * ")" * *
      volume empty$
        { "there's a number but no volume in " cite$ * warning$ }
        'skip$
      if$
    }
  if$  
}

FUNCTION {format.series}
{ series empty$
    {""} 
    { #0 bother =
        {booktitle empty$ 
                {"\textit{" series * "}" *}
                {series}
         if$       
        }
        {"\bsertitle{" series * "}" * }
      if$
    }
  if$
}

FUNCTION {format.ser.vol}
{ series empty$
    'skip$ 
    { #0 bother =
        {"in \textit{" series * "}" * output}
        {booktitle empty$
                {"in \bbtitle{" series * "}" * output}
                {"in \bsertitle{" series * "}" * output}
         if$    
        }
      if$
      volume empty$
          'skip$
          {insert.comma format.volume output}
      if$ 
    }
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { "" 'v :=
      "" 'l :=    
      "" 'f :=    
      edition "l" change.case$ 's :=
         "1" s =
         "first" s =        
         or
           {
             "1" 'v :=
            "st" 'l :=  
           }
           {
             "2" s =
             "second" s =       
             or
               {
                 "2" 'v :=
                "nd" 'l :=  
               }
               {
                 "3" s =
                 "third" s =        
                 or
                   {
                     "3" 'v :=
                    "rd" 'l :=  
                   }
                   {
                     "4" s =
                     "fourth" s =       
                     or
                       {
                         "4" 'v :=
                        "th" 'l :=  
                       }
                       {
                         "5" s =
                         "fifth" s =        
                         or
                           {
                             "5" 'v :=
                            "th" 'l :=  
                           }
                           {
                             "6" s =
                             "sixth" s =        
                             or
                               {
                                 "6" 'v :=
                                "th" 'l :=  
                               }
                               {
                                 "7" s =
                                 "seventh" s =        
                                 or
                                   {
                                     "7" 'v :=
                                    "th" 'l :=  
                                   }
                                   {
                                     "8" s =
                                     "eighth" s =       
                                     or
                                       {
                                         "8" 'v :=
                                        "th" 'l :=  
                                       }
                                       {
                                         "9" s =
                                         "nineth" s =       
                                         or
                                           {
                                             "9" 'v :=
                                            "th" 'l :=  
                                           }
                                           {
                                               edition "t" change.case$ 'f :=
                                           }
                                         if$   
                                       }
                                     if$   
                                   }
                                 if$   
                               }
                             if$   
                           }
                         if$   
                       }
                     if$   
                   }
                 if$   
               }
             if$   
           }
         if$   
        #0 bother =   
           {  f "" =
               { v l *  'f :=}
               'skip$
             if$
           }
           {  f "" =
               { "\bedition{" v * "}" * l * 'f :=}
               { "\bedition{" f * "}" * 'f :=}
             if$
           }
       if$
       f " edn" *
    }
  if$
}

FUNCTION {format.isbn}
{ isbn empty$
    { "" }
    { "ISBN " isbn "isbn" make.tag * }  
  if$
}
FUNCTION {format.issn}
{ issn empty$
    { "" }
    { "ISSN " issn * "comment" make.tag  }  
  if$
}

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    "" 'l :=

    {t empty$ not}
    { t #1 #1 substring$  's :=
        
      s is.num not
        { "" l = not
                {t find.integer 
                                {#1 'multiresult := }
                                {#0 'multiresult := }
                        if$
                        "" 't :=
                   }
                   {
                   t #2 global.max$ substring$ 't :=
                   }
               if$  
        }
        { l  s * 'l :=
            t #2 global.max$ substring$ 't := 
        }
      if$
    }
  while$
  multiresult 
}

FUNCTION {do.pages}
{'t :=
    "" 'j :=
    "" 'v :=
    {t empty$ not}
    { t #1 #1 substring$  's :=
      "-" s =
        { t #1 #2 substring$ "--" =
            {t #3 global.max$ substring$ 'v := }
            {t #2 global.max$ substring$ 'v := }
          if$
          "" 't :=
        }
        {	j  s * 'j :=
              t #2 global.max$ substring$ 't := 
        }
      if$
    }
  while$
  
  % remove " " at first page end
  j j text.length$ #1   substring$ " " =
    {j #1 j text.length$ #1 -   substring$ 'j :=}
    'skip$
  if$
  
  % remove " " at begin of last page
  v #1 #1 substring$ " " = 
    {v #2 global.max$ substring$ 'v :=}
    'skip$
  if$
  j  "fpage" make.tag "--"  * v "lpage" make.tag *
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
        {type$ "article" =
            {""}
            {"pp.~"} 
         if$  
            pages do.pages *
        }
        {type$ "article" =
            {""}
            {"p.~"} 
         if$  
            pages "fpage" make.tag *
        }

      if$
    }
  if$
}

FUNCTION {replace.tilde}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "~" =
            {   { t #1 #1 substring$ "~" = }
                { "\texttildelow " *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {string.in.url}{
  'temp.str :=
  url 't :=
  #0 'return :=
  #0 'done :=
  temp.str text.length$ 'str.length :=
  str.length #0 >
    { 
      { done not
        t empty$ not
        and
      }
      { t #1 str.length substring$ temp.str =
          { #1 'return := 
            #1 'done := 
          }
          { t text.length$ str.length = 
              {#1 'done := }
              {t #2 global.max$ substring$ 't := }
            if$
          }
        if$
      }
    while$
    }
    'skip$
  if$   
  return 
}


FUNCTION {doi.in.url}{
  doi empty$
    {#0}
    {doi string.in.url}
  if$
}

FUNCTION {format.url}
{ url empty$
    { "" }
    {
      doi.in.url #1 =
            {""}
            {"\url{" url replace.tilde * "}" * }
      if$
     %url top$   
    } 
  if$
}



FUNCTION {format.doi}
{ doi empty$
        {""}
        {
         "nodoi" is.in.list
       {""}
         {"doi:\doiurl{" doi * "}" *}
     if$
        }
  if$
}

FUNCTION {format.in.ed.booktitle}
{ booktitle empty$
    { 
        format.edition output
        edition empty$
            'insert.comma
          'skip$
      if$ 
      format.middle.editors 
    }
    {"in: " booktitle "btitle" make.tag * output
        series empty$ 
            {volume empty$
                'skip$
                'insert.comma 
             if$
             format.volume output
            }
             'skip$
         if$  
         edition empty$
             'skip$ 
              'insert.comma
          if$ 
          format.edition output

        editor empty$
          'skip$
          {
            edition empty$
                'insert.comma
                  'skip$
            if$ 
            format.middle.editors  
          }
        if$
    }    
  if$
}

FUNCTION {format.in.ed}
{ "notnumber" last.label =
          {insert.comma  format.middle.editors }
            {
                series empty$
                        'skip$
                { 
                    "in: "  series "btitle" make.tag * output
                    volume empty$
                        'skip$
                        {insert.comma format.volume output}
                    if$ 
                }
            if$
                editor empty$ author empty$ or
                        'skip$          
                {insert.comma format.middle.editors}
        if$    
            } 
    if$
}

FUNCTION {format.tr.number}
{ type empty$
    { "Technical Report" }
    'type
  if$
  number empty$
    'skip$
    {", " number *  *}
  if$
}

FUNCTION {patent.number}
{    number empty$
         'skip$
    { number }
  if$
}

INTEGERS { len }

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}   
 
FUNCTION {format.inpres}
{ "l" change.case$ 't :=
      "in press: " #10
       "in press. " #10
        "in press " #9 t chop.word
      chop.word
     chop.word
  #1 global.max$ substring$
}

FUNCTION {bcomment.note}
{note empty$
'skip$
 {note format.inpres "comment" make.tag output}
 if$
}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}

FUNCTION {format.chapter.and.note}
{   note empty$ chapter empty$ organization empty$ and and
        'skip$
    { chapter empty$
        "notnumber" last.label =
        or
            {organization.outputed #1 = 
             organization empty$
             type$ "manual" =
             or or 
                    {note "comment" make.tag output}
                    {note empty$
                            {organization "comment" make.tag output}
                            {organization ". " * note * "comment" make.tag output}
                     if$
                    }
             if$
            }
            {note empty$
                {"Chapter~"  chapter * "comment" make.tag output}
                        {   type empty$
                                { "Chapter~" }
                                { type "t" change.case$ "Section" = 
                                         {  "Sect. " }
                                        { "Chapter~" }
                                    if$ 
                                }
                            if$
                            chapter * ". " * note * "comment" make.tag  output
                        }
                    if$
                }
            if$
    }
    if$
        
}

FUNCTION {item.end}
{
  insert.comma
  type$ "unpublished" =
    { note empty$
            'skip$
            {note.presented
                    'skip$
                    {note output}
             if$    
            }
        if$ 
    }
    'format.chapter.and.note 
  if$
  type$ "article" =
      'skip$
      {
        new.sentence 
        format.issn output
      }
  if$
  new.sentence
  format.isbn output
  new.sentence
  format.doi  output
  new.sentence
  format.url output
  fin.entry
}

FUNCTION {insert.element}{
  bother #0 =
    {element "{botherref}" * write$ newline$ }
    {
         type$ "article" =
            {element "{barticle}" * write$ newline$}
            'skip$
         if$
         type$ "book" =
         type$ "proceedings" =
         type$ "manual" =
         type$ "booklet" =
         or or or 
          {element "{bbook}" * write$ newline$}
          'skip$
         if$
         type$ "inbook" =
                { "notnumber" last.label =
                        {
                            author empty$
                                {element "{bbook}" * write$ newline$}
                                {element "{bchapter}" * write$ newline$}
                            if$ 
                        }
                        { author empty$ editor empty$ or
                                {element "{bbook}" * write$ newline$}
                                {element "{bchapter}" * write$ newline$}
                            if$ 
                        }
                    if$ 
                }
                  'skip$
           if$
      type$ "incollection" =
      type$ "inproceedings" =
      type$ "conference" =
      or or
        {element "{bchapter}" * write$ newline$}
        'skip$
      if$

    }
  if$
}

FUNCTION {end.element}
{ "\end" 'element :=
    item.end
  insert.element
  "\endbibitem" write$ newline$
}

FUNCTION {begin.element}
{ "\begin" 'element :=
  #0 'organization.outputed :=
  insert.element
}

FUNCTION {assign.global.values}{

  %%%%%%%%%% Default values: #1->on, #0->off  
  info empty$
    'skip$
    { info "1" =
          {#1 'default.info :=} 
          {#0 'default.info :=}
      if$
        }
    if$
}
FUNCTION {prin.info}{
    default.info
        {
            "alpha" is.in.list
                {"alpha"}
                {
                    "nameyear" is.in.list
                        {"nameyear"}
                        {"number"}
                    if$
                }
            if$
            *
            ", sorted="
            * 
            "unsort" is.in.list
                {"no"}
                {"yes"}
            if$
            *

            "nodoi" is.in.list
                {", other options=nodoi" *}
                'skip$
            if$
            write$ newline$
        }
        'skip$
    if$ 
}

FUNCTION {settings}
{
  set.settings not
        { 
          assign.global.values
          #1 'set.settings :=
          prin.info
        }
        'skip$
    if$
}

FUNCTION {article}
{ output.bibitem
  author empty$
  institution empty$ 
  editor empty$
  and and
  journal empty$ 
  year empty$ 
  or or 
    {#0 'bother :=} 
    { 
      volume empty$ pages  empty$ 
      or 
        {
          volume empty$ pages  empty$ doi empty$ not
          and and   
            {#1 'bother :=}
            {#0 'bother :=}
          if$ 
        } 
        {#1 'bother :=}
      if$       
    } 
  if$
  begin.element
  author.or.editor
  format.title  "atitle" make.tag output
  journal empty$
      'skip$
      {insert.comma journal  "jtitle" make.tag output}
  if$
  volume empty$
      'skip$
      {format.art.vol output }
  if$
  format.article.date output
  pages empty$
      'skip$
      'insert.comma
  if$
  format.pages output
  end.element
}

FUNCTION {patent}{ 
  output.bibitem
  #0 'bother :=
  begin.element
  author.or.editor
  format.title "title" output.check
  patent.number output
  insert.comma
  formatpatent.date output
  end.element
}

FUNCTION { other } { patent }

FUNCTION {book}
{ output.bibitem
  type$ "manual" =
  type$ "booklet" =
  type$ "book" =
  or or
    {
      author empty$
      institution empty$ 
      organization empty$ 
      editor empty$
      and and and
      title empty$ 
      year empty$ 
      or or 
        {#0 'bother :=} 
        {#1 'bother :=} 
      if$ 
    }
    {
      #0 'bother :=
    }
  if$    
  begin.element
  author.or.editor.or.organization
  
  type$ "manual" =
  type$ "booklet" =
  type$ "book" =
  or or
    {
      title empty$
        'skip$
        {format.title  "btitle" make.tag "title" output.check}
      if$
    }
    {format.title output}
  if$
  insert.comma
  series empty$
    {
     volume empty$
      'skip$
      'insert.comma
     if$    
     format.volume output
     edition empty$
      'skip$
      'insert.comma 
     if$
     format.edition output
    }
    {
     insert.comma
     format.edition output
     insert.comma
     format.series output
     volume empty$
      'skip$
      'insert.comma
     if$    
     format.volume output
    }
  if$ 
  insert.comma
  format.pub.address
  insert.comma
  format.pages output
  end.element      
}
FUNCTION {manual}{book}
FUNCTION {booklet}{book}
FUNCTION {misc}{book}
FUNCTION {unpublished}{book}
FUNCTION {default.type} { book }

FUNCTION {incollection}
{ output.bibitem
  author empty$
  editor empty$
  and
  booktitle empty$ 
  title empty$
  and
  year empty$ 
  or or
    {#0 'bother :=} 
    {#1 'bother :=} 
  if$ 
  begin.element
  author.or.editor
  format.title
  type$ "inbook" = 
      {"btitle"}
      {"ctitle"}
  if$
  make.tag "title" output.check
  title empty$ 
    'skip$
    {insert.comma}
  if$
  booktitle empty$
    'format.ser.vol
    'skip$
  if$
  insert.comma
  format.in.ed.booktitle 
  insert.comma
  series empty$
    'skip$
    {
      booktitle empty$
        'skip$
        { 
          format.series output
          volume empty$
            'skip$
            'insert.comma
          if$ 
          format.volume output
        }
      if$  
    }
  if$ 
  insert.comma
  format.pub.address
  insert.comma
  format.pages output
  end.element
}
FUNCTION {inproceedings}{incollection}
FUNCTION {proceedings}{incollection}
FUNCTION {conference}{incollection} 
FUNCTION {inbook}{incollection}

FUNCTION {phdthesis}
{output.bibitem
  #0 'bother :=
  begin.element
  author.or.editor
  format.title "title" output.check
  insert.comma
  type$ "mastersthesis"  =
    {"Master's thesis" format.thesis.type output.nonnull}
    {"PhD thesis" format.thesis.type output.nonnull}
  if$   
  school empty$
    'skip$
    { insert.comma }
  if$
  school "school" output.check
  address empty$
    'skip$
    { insert.comma address output}
  if$
  year empty$
    'skip$
    { insert.comma year output}
  if$
  end.element
}

FUNCTION {mastersthesis}{phdthesis}

FUNCTION {techreport}
{output.bibitem
  #0 'bother :=
  begin.element
  author.or.editor
  format.title "title" output.check
  insert.comma
  format.tr.number output.nonnull
  institution empty$
    'skip$
    { insert.comma }
  if$
  institution "institution" output.check
  address empty$
    'skip$
    { insert.comma }
  if$
  address output
  year empty$
    'skip$
    { insert.comma year output}
  if$
  end.element
}

MACRO {jan} {"January"}
MACRO {feb} {"February"}
MACRO {mar} {"March"}
MACRO {apr} {"April"}
MACRO {may} {"May"}
MACRO {jun} {"June"}
MACRO {jul} {"July"}
MACRO {aug} {"August"}
MACRO {sep} {"September"}
MACRO {oct} {"October"}
MACRO {nov} {"November"}
MACRO {dec} {"December"}
READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

INTEGERS { et.al.char.used }

FUNCTION {initialize.et.al.char.used}
{ #0 'et.al.char.used :=
}

%%%%% setting default options
FUNCTION {set.default.opt}
{"sort" 'list.string :=}

EXECUTE {set.default.opt}

FUNCTION {assign.opt}{
  % First need to assign options 
  "settings" type$ =
      {options empty$
          'skip$
          {options 'list.string :=}
       if$
      }
      'skip$
  if$
}

ITERATE {assign.opt}
EXECUTE {initialize.et.al.char.used}

FUNCTION {alpha.format.lab.names}
{ 's :=
  s num.names$ 'numnames :=
  numnames #1 >
    { numnames #4 >
    { #3 'namesleft := }
    { numnames 'namesleft := }
      if$
      #1 'nameptr :=
      ""
    { namesleft #0 > }
    { nameptr numnames =
        { s nameptr "{ff }{vv }{ll}{ jj}" format.name$ "others" =
        { "+" *
          #1 'et.al.char.used :=
        }
        { s nameptr "{v{}}{l{}}" format.name$ * }
          if$
        }
        { s nameptr "{v{}}{l{}}" format.name$ * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
      while$
      numnames #4 >
    { "+" *
      #1 'et.al.char.used :=
    }
    'skip$
      if$
    }
    { s #1 "{v{}}{l{}}" format.name$
      duplicate$ text.length$ #2 <
    { pop$ s #1 "{ll}" format.name$ #3 text.prefix$ }
    'skip$
      if$
    }
  if$
}

FUNCTION {standart.format.lab.names}
{ 's :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ duplicate$
  #2 >
    { pop$ " et~al." * }
    { #2 <
        'skip$
        { s #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            { " et~al." * }
            { " and " * s #2 "{vv~}{ll}" format.name$ * }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.lab.names}{
    "alpha" is.in.list
        'alpha.format.lab.names
      'standart.format.lab.names
    if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
    { cite$ #1 #3 substring$ }
    { key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
    { key empty$
        { cite$ #1 #3 substring$ }
%        { "zzz" }
        { key #3 text.prefix$ }
      if$
    }
    { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.key.organization.label}
{ author empty$
    { key empty$
    { organization empty$
         { cite$ #1 #3 substring$ }
%        { "zzz" }
        { "The " #4 organization chop.word #3 text.prefix$ }
      if$
    }
    { key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.organization.label}
{ editor empty$
    { key empty$
    { organization empty$
        { cite$ #1 #3 substring$ }
%        { "zzz" }
        { "The " #4 organization chop.word #3 text.prefix$ }
      if$
    }
    { key #3 text.prefix$ }
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.short.authors}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
        'editor.key.organization.label
        { type$ "manual" =
            'author.key.organization.label
            'author.key.label
          if$
        }
      if$
    }
  if$
  'short.list :=
}

FUNCTION {calc.label}
{ calc.short.authors
  short.list

  "alpha" is.in.list
  {
    duplicate$
    year field.or.null purify$ #-1 #2 substring$
    *
    'label :=
    year field.or.null purify$ #-1 #4 substring$
    *
    sortify 'sort.label :=
  }
  {
    "("
    *
    year duplicate$ empty$
    short.list key field.or.null = or
      { pop$ "" }
      'skip$
    if$
    *
    'label :=
    label 'sort.label :=
  } 
  if$
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
        { "   " * 
          "alpha" is.in.list
            'skip$
            {numnames int.to.str$ * "   " *}
          if$
        }
        'skip$
      if$
      s nameptr "{vv{ } }{ll{ }}{  ff{ }}{  jj{ }}" format.name$ 't :=
      nameptr numnames = 
      t "others" = 
      and
        { "\betal" * }
        { t sortify * }
      if$
        #3 numnames  <
        "alpha" is.in.list not
        and
              {#0 'namesleft :=
                       " zzz   " *
              }    
          {
               nameptr #1 + 'nameptr :=
              namesleft #1 - 'namesleft :=
          }
        if$   
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
        { "to sort, need author or key in " cite$ * warning$
          "zzz"
        }
        { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { key empty$
            { "to sort, need author, editor, or key in " cite$ * warning$
              "z"
            }
            { key sortify }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
        { key empty$
            { "to sort, need author, organization, or key in " cite$ * warning$
              "z"
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
        { key empty$
            { "to sort, need editor, organization, or key in " cite$ * warning$
              "z"
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

FUNCTION {presort}
{ calc.label
  sort.label sortify
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  or
    { author empty$
        'editor.organization.sort
        'author.organization.sort
      if$
    }
    { type$ "proceedings" =
      type$ "incollection" =
       or
            {author empty$
                'editor.organization.sort
            'author.organization.sort
        if$
        }
        { type$ "manual" =
            'author.organization.sort
            'author.sort
          if$
        }
      if$
    }
  if$
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$ 'sort.label :=
  sort.label *
  #1 entry.max$ substring$ 'sort.key$ :=
  
  "settings" type$ =
      {"aaa"}
      {"unsort" is.in.list
        {"bb"}
        {"alpha" is.in.list
            {sort.key$}
            {sort.label}
          if$  
           }
       if$
      }
      if$
  'sort.key$ :=
}


ITERATE {presort}

SORT

INTEGERS { longest.label.width last.extra.num number.label }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #0 int.to.chr$ 'last.label :=
  "" 'j :=
  #0 'longest.label.width :=
  #0 'last.extra.num :=
  #0 'number.label :=
}

FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
  number.label #1 + 'number.label :=
}

FUNCTION {reverse.pass}
{ j "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  extra.label 'j :=
  label extra.label * 'label :=
}

EXECUTE {initialize.longest.label}

ITERATE {forward.pass}

REVERSE {reverse.pass}

FUNCTION {write.preambule}
{ "\begin{thebibliography}{" number.label int.to.str$ * "}" * write$ newline$
    "% BibTex style file: ios1.bst, 2019-01-11" write$ newline$
    "\ifx \bisbn   \undefined \def \bisbn  #1{ISBN #1}\fi" write$ newline$
    "\ifx \binits  \undefined \def \binits#1{#1} \fi" write$ newline$
    "\ifx \bauthor  \undefined \def \bauthor#1{#1} \fi" write$ newline$
    "\ifx \bjtitle  \undefined \def \bjtitle#1{\textit{#1}}\fi" write$ newline$
    "\ifx \batitle  \undefined \def \batitle#1{#1} \fi" write$ newline$
    "\ifx \bctitle  \undefined \def \bctitle#1{#1} \fi" write$ newline$
    "\ifx \bvolume  \undefined \def \bvolume#1{\textbf{#1}}\fi" write$ newline$
    "\ifx \byear  \undefined \def \byear#1{#1} \fi" write$ newline$
    "\ifx \bissue  \undefined \def \bissue#1{#1} \fi" write$ newline$
    "\ifx \bfpage  \undefined \def \bfpage#1{#1} \fi" write$ newline$
    "\ifx \blpage  \undefined \def \blpage #1{#1} \fi" write$ newline$
    "\ifx \burl  \undefined \def \burl#1{#1} \fi" write$ newline$
    "\ifx \doiurl  \undefined \def \doiurl#1{#1} \fi" write$ newline$
    "\ifx \betal  \undefined \def \betal{et al.} \fi" write$ newline$
    "\ifx \binstitute  \undefined \def \binstitute#1{#1} \fi" write$ newline$
    "\ifx \beditor  \undefined \def \beditor#1{#1} \fi" write$ newline$
    "\ifx \bpublisher  \undefined \def \bpublisher#1{#1} \fi" write$ newline$
    "\ifx \bbtitle  \undefined \def \bbtitle#1{\textit{#1}} \fi" write$ newline$
    "\ifx \bedition  \undefined \def \bedition#1{#1} \fi" write$ newline$
    "\ifx \bseriesno  \undefined \def \bseriesno#1{#1} \fi" write$ newline$
    "\ifx \blocation  \undefined \def \blocation#1{#1} \fi" write$ newline$
    "\ifx \bsertitle  \undefined \def \bsertitle#1{#1} \fi" write$ newline$
    "\ifx \bsnm \undefined \def \bsnm#1{#1} \fi" write$ newline$
    "\ifx \bsuffix \undefined \def \bsuffix#1{#1} \fi" write$ newline$
    "\ifx \bparticle \undefined \def \bparticle#1{#1} \fi" write$ newline$
    "\ifx \barticle \undefined \def \barticle#1{#1} \fi" write$ newline$  
}


FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  write.preambule
  "\ifx \botherref \undefined \def \botherref #1{#1} \fi" write$ newline$
  "\ifx \url \undefined \def \url#1{#1} \fi" write$ newline$
    "\ifx \bchapter \undefined \def \bchapter#1{#1} \fi" write$ newline$
    "\ifx \bbook \undefined \def \bbook#1{#1} \fi" write$ newline$
    "\ifx \bcomment \undefined \def \bcomment#1{#1} \fi" write$ newline$
    "\ifx \oauthor \undefined \def \oauthor#1{#1} \fi" write$ newline$
    "\ifx \citeauthoryear \undefined \def \citeauthoryear#1{#1} \fi" write$ newline$
    "\ifx \texttildelow  \undefined \def \texttildelow{\symbol{126}} \fi" write$ newline$
    "\def \endbibitem {}" write$ newline$
    "\ifx \bconflocation  \undefined \def \bconflocation#1{#1} \fi" write$ newline$
}
EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
 "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}




