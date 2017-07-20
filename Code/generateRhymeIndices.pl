#!/usr/bin/perl 
use strict;
use warnings;

use List::MoreUtils qw/ uniq /;



my ($oina, $azkenOina, $errima, @hitzak, @puntuak, $puntu1, $puntu2, $puntu3, $lerro, $puntua, $poto);
my @errimak;
my @errimakindices;
my $cont;
# ERRIMA BILATZAILEA
sub errima_patroia($) {
	my $errima = shift;
	my $patroia;	

	# ERRIMA PATROIA OSATU

	# B-D-G-R
	$patroia = $1 . $2 .'[bdgr]' .$3 if ($errima =~ m/([aeiou])?([lnr])?[bdgr]([aeiou])\b/i);

	# P-T-K
	$patroia = $1 . '[ptk]' . $2 if ($errima =~ m/([aeiou]?)[ptk]([aeiou])\b/i);
	$patroia = $1 . '[ptk]' . $2 if ($errima =~ m/([lnr])[ptk]([aeiou])\b/i);
	$patroia = $1 . $2 . '[ptk]' if ($errima =~ m/([aeiou])(r)?[ptk]\b/i);
	$patroia = 't[sz]a[tk]' if ($errima =~ m/t[sz]a[tk]\b/i);
	$patroia = $1 . 'i[tk]' if ($errima =~ m/([aeo])i[tk]\b/i);
	$patroia = 'os[tk]' if ($errima =~ m/os[tk]\b/i);
	$patroia = $1 . '[szx][ptk]' . $2 if ($errima =~ m/([aeiou])[szx][ptk]([aeiou])\b/i);

	# M-N
	$patroia = '([bdgrflnm]|rr)' . $2 .'n' if ($errima =~ m/([bdgrflnmph]|rr)([aiu])n\b/i);
	$patroia = $1 . '([bdgrflnm]|rr)' . $3 .'n' if ($errima =~ m/([aeiou])([bdgrflnmph]|rr)([aiu])n\b/i);
	$patroia = $1 . 'an' if ($errima =~ m/([eiou])an\b/i);
	$patroia = $1 . 'en' if ($errima =~ m/([eiou])en\b/i);
	$patroia = $1 . '[sz]ean' if ($errima =~ m/(t)?[sz]ean\b/i);
	$patroia = $1 . '[ptk]' . $2 . 'n' if ($errima =~ m/([aeiou])[ptk]([aei])n\b/i);
	$patroia = '[sz][ptk]' . $1 .'n' if ($errima =~ m/[sz][ptk]([aeiu])n\b/i);
	$patroia =  $1 . '[sz]' . $2 . 'n' if ($errima =~ m/([aeiou])[sz]([eau])n\b/i);
	$patroia = 't[sz]' . $1 . 'n' if ($errima =~ m/t[sz]([eau])n\b/i);	
	$patroia = $1 . '[bdgrflnm]' . 'en' if ($errima =~ m/([aeiou])[bdgrflnm]en\b/i);
	$patroia = 'rren' if ($errima =~ m/rren\b/i);
	$patroia = 'r[bdgflnm]' . 'en' if ($errima =~ m/r[bdgflnm]en\b/i);
	$patroia = '[szx]' . $1 . 'n' if ($errima =~ m/[szx]([eiou])n\b/i);
	$patroia = 'r[kpt]en' if ($errima =~ m/r[kpt]en\b/i);
	$patroia = $1 . 'in' if ($errima =~ m/([aeou])in\b/i);
	$patroia = $1 . '[bdgrflnm]?' . 'in' if ($errima =~ m/([aeiou])([bdgrflnm])?in\b/i);
	$patroia = $1 . '[bdgrflnm]' . 'on' if ($errima =~ m/([aeiou])([bdgrflnm])on\b/i);
	$patroia = '[ptk]' . $1 .'n' if ($errima =~ m/[ptk]([ou])n\b/i);
	$patroia = $1 . 'un' if ($errima =~ m/([ae])un\b/i);
	$patroia = $1 . '[mn]' . $2 if ($errima =~ m/([aeiou])[mn]([aeiou])\b/i);
	$patroia = $1 . '([aeou])ina' . $2 if ($errima =~ m/([aeou])ina\b/i);
	$patroia = $1 . '([mn]a)' if ($errima =~ m/([sz]|r)([mn])a\b/i);
	$patroia = 'aino' if ($errima =~ m/aino\b/i);

	# L
	$patroia = $1 . '([bdgrflnmhkptsz])*' . $3 .'l' 
		if ($errima =~ m/([aeiou])?([bdgrflnmhkptsz])*([aeiou])l\b/i);
	$patroia = $1 . 'l' . $2 if ($errima =~ m/([aeiou])l([aeiou])\b/i);

	# RR
	$patroia = $1 . '([bdgrflnmhkptsz])*' . $3 .'r' 
		if ($errima =~ m/([aeiou])([bdgrflnmhkptsz])*([aeiou])r\b/i);
	$patroia = '([szx])([ptk])er' if ($errima =~ m/([szx])([ptk])er\b/i);
	$patroia = $1 . 'rr' . $2  if ($errima =~ m/([aeiou])rr([aeiou])\b/i);

	# S-Z
	$patroia = 'ez' if ($errima =~ m/ez\b/i);
	$patroia = $1 . '([bdgrflnmhkptsz])+' . $3 .'[sz]' 
		if ($errima =~ m/([aeiou])([bdgrflnmhkptsz]?)+([aeiou])[sz]\b/i);
	$patroia = '[ptk]' . $1 .'[sz]' if ($errima =~ m/[ptk]([ae])[sz]\b/i);
	$patroia = 'rrez'  if ($errima =~ m/rrez\b/i);
	$patroia = '[aeo]h?i[sz]'  if ($errima =~ m/[aeo]h?i[sz]\b/i);
	$patroia = 'au[sz]'  if ($errima =~ m/au[sz]\b/i);
	$patroia = $1 . '[szx]' . $2  if ($errima =~ m/([aeiou])[szx]([aeiou])\b/i);

	# TS-TZ-TX
	$patroia = $1 . 't[szx]' if ($errima =~ m/([aeiou])t[szx]\b/i);
	$patroia = $1 . 'itz' if ($errima =~ m/([ao])itz\b/i);
	$patroia = 'au(n)?t[sz]'  if ($errima =~ m/au(n)?t[sz]\b/i);
	$patroia = $1 . 't[szx]' . $2 if ($errima =~ m/([aeiou])t[szx]([aeiou])\b/i); 
	$patroia = $1 . 'nt[szx]a' if ($errima =~ m/([aeiou])nt[szx]a\b/i);
	$patroia = $1 . 't[szx]' if ($errima =~ m/([rnl])t[szx]\b/i);
	 

	#DIPTONGOAK
	$patroia = $1 . '([bdgrflnmhkpt])+' . '[szx]?' . $3 . 'a' 
		if ($errima =~ m/([aeiou])([bdgrflnmhkpt])+[szx]?([ei])a\b/i);
	$patroia = $1 . 'ia' if ($errima =~ m/([ae])ia\b/i);
	$patroia = $1 . '[szx]ia' if ($errima =~ m/([aeiou])[szx]ia\b/i);
	$patroia = $1 . '[sxz]?' . 't[szx]' . $2 . 'a' if ($errima =~ m/([aeiou])[sxz]?t[szx]([ei])a\b/i);
	$patroia = $1 . '[kpt]ia' if ($errima =~ m/([aeiou])[kpt]ia\b/i);
	$patroia = $1 . '[bdgrnmhkpt]oa' if ($errima =~ m/([aeiou]?)[bdgrslnmhkpt]+oa\b/i);
	$patroia = '[bdgrnmhkpt]ua' if ($errima =~ m/[bdgrslnmhkpt]+ua\b/i);
	$patroia = $1 . 'h?' . $2 if ($errima =~ m/([aeio])h?([eiu])\b/i);
	$patroia = $1 . 'o' if ($errima =~ m/([aei])o\b/i);
	$patroia = $1 . 'io' if ($errima =~ m/([ae])io\b/i);
	$patroia = '[sz]io' if ($errima =~ m/[sz]io\b/i);
	$patroia = $1 . 'u' if ($errima =~ m/([aei])u\b/i);

	$patroia = 'nik' if ($errima =~ m/nik\b/i);
	$patroia = 'rik' if ($errima =~ m/rik\b/i);
	$patroia = 'rrik' if ($errima =~ m/rrik\b/i);
	$patroia = 'lik' if ($errima =~ m/lik\b/i);
	$patroia = '[sz]ik' if ($errima =~ m/[sz]ik\b/i);
	$patroia = 't[sz]ik' if ($errima =~ m/t[sz]ik\b/i);

	# Otzeta
	$patroia = 'zu' if ($errima =~ m/zu\b/i);
	$patroia = 'su' if ($errima =~ m/su\b/i);
	$patroia = 'tsu' if ($errima =~ m/tsu\b/i);
	$patroia = 'est' if ($errima =~ m/est\b/i);
	$patroia = 'oz' if ($errima =~ m/oz\b/i);
	$patroia = 'ar' if ($errima =~ m/ar\b/i);
	$patroia = 'ion' if ($errima =~ m/ion\b/i);
	$patroia = 'ea' if ($errima =~ m/ea\b/i);
	$patroia = 'oia' if ($errima =~ m/oia\b/i);
	$patroia = 'tia' if ($errima =~ m/tia\b/i);
	$patroia = 'ioa' if ($errima =~ m/ioa\b/i);
	$patroia = 'la' if ($errima =~ m/la\b/i);
	$patroia = 'zoa' if ($errima =~ m/zoa\b/i);
	$patroia = 'xoa' if ($errima =~ m/xoa\b/i);
	$patroia = 'ua' if ($errima =~ m/ua\b/i);
	$patroia = 'tsa' if ($errima =~ m/tsa\b/i);
	$patroia = 'zue' if ($errima =~ m/zue\b/i);
	$patroia = 'an' if ($errima =~ m/an\b/i);
	$patroia = 'txe' if ($errima =~ m/txe\b/i);
	$patroia = 'ne' if ($errima =~ m/ne\b/i);
	$patroia = 'le' if ($errima =~ m/le\b/i);
	$patroia = 'en' if ($errima =~ m/en\b/i);
	$patroia = 'ia' if ($errima =~ m/ia\b/i);
	$patroia = 'ma' if ($errima =~ m/ma\b/i);
	$patroia = 'eoa' if ($errima =~ m/eoa\b/i);
	$patroia = 'za' if ($errima =~ m/za\b/i);
	$patroia = 'tza' if ($errima =~ m/tza\b/i);
	$patroia = 'tsi' if ($errima =~ m/tsi\b/i);
	$patroia = 'tzi' if ($errima =~ m/tzi\b/i);
	$patroia = 'on' if ($errima =~ m/on\b/i);
	$patroia = 'jo' if ($errima =~ m/jo\b/i);
	$patroia = 'tso' if ($errima =~ m/tso\b/i);
	$patroia = 'tzo' if ($errima =~ m/tzo\b/i);
	$patroia = 'txo' if ($errima =~ m/txo\b/i);
	$patroia = 'or' if ($errima =~ m/or\b/i);
	$patroia = 'in' if ($errima =~ m/in\b/i);
	$patroia = 'txa' if ($errima =~ m/txa\b/i);
	return $patroia;
}

# Azken puntua
# open(FITXPUNT,$ARGV[0]) or die("Errorea! Ezin fitxategia zabaldu\n");
# my $puntuLer = <FITXPUNT>;
# chomp($puntuLer);

my $puntuLer = $ARGV[0];

open(FITX,$ARGV[1]) or die("Errorea! Ezin fitxategia zabaldu\n");
@puntuak = <FITX>;
close(FITX);
# $puntua = $puntuak[$puntuLer];
$puntua = $ARGV[0];
chomp($puntua);	
push(@errimak, $puntua);
@hitzak = split(/[\n\s-]/, $puntua);
$azkenOina = pop(@hitzak);
$errima = errima_patroia($azkenOina);
#print ("$errima\n\n");

#if (length $errima)
#{
$cont = 0;
	foreach $lerro(@puntuak) {
		$cont++;
		chomp($lerro);	
		@hitzak = split(/[\n\s-]/, $lerro);
		$oina = pop(@hitzak);
		if ($oina =~ m/$errima$/) {
			push(@errimak, $lerro);
			push(@errimakindices, $cont);
		}
	}
	foreach ( @errimakindices ) {
	    print $_, "\n";
	}
#}
=pod
else
{
	open(FITX2, '>>', "noRhyme.txt");
	print FITX2 $azkenOina;
	print FITX2 "\n";
	close(FITX2);
}
=cut
