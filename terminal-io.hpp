auto colorize_str(const std::string& srcstr, QTeXgrcolor c) -> std::string {
  std::string ansipre = "\033[";
  using namespace QTeXgrcolors;
  if       (c == red)       ansipre += "0;31";
    else if(c == green)     ansipre += "0;32";
    else if(c == blue)      ansipre += "0;34";
    else if(c == i_red)     ansipre += "1;31";
    else if(c == i_green)   ansipre += "1;32";
    else if(c == i_blue)    ansipre += "1;34";
    else if(c == teal)      ansipre += "0;36";
    else if(c == lilac)     ansipre += "0;35";
    else if(c == brown)     ansipre += "0;33";
    else if(c == i_teal)    ansipre += "1;36";
    else if(c == i_lilac)   ansipre += "1;35";
    else if(c == i_brown)   ansipre += "1;33";
    else if(c == grey)      ansipre += "0;37";
    else if(c == i_grey)    ansipre += "1;37";
    else if(c == constrast) ansipre += "0;31";
    else if(c == neutral)   ansipre += "0;30";
    else                    return srcstr;
  ansipre += "m";
  
  return ansipre + srcstr + "\033[0m";
}