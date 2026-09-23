#!/bin/sh

set -euo pipefail
IFS=$'\n\t'

if [ $# -lt 3 ]; then
    echo "usage:" $0 "<output-path>" "<test-input-path>" "<..command..>"
    exit
fi

networkpath=$1
bestnetworkpath="${1}.best"
echo "saving best in" $bestnetworkpath

testlosses="${1}.test_losses"
echo "saving test losses in" $testlosses

testinput=$2
echo "loading test inputs from" $testinput

shift 2

while true
do
    $@

    $1 --mode test --path $networkpath --input $testinput | grep --only-matching '[\.0-9]*' >> $testlosses

    if perl -e 'use warnings; use strict; use List::Util qw( min ); my @losses = (); while(<>) { chomp($_); next if($_ eq ""); push(@losses, $_) }; if (scalar(@losses) < 2) { exit 0 }; my @reverselosses = (reverse(@losses))[0..(min(9, scalar(@losses) - 1))]; my @sorted = sort {$a <=> $b} @reverselosses; if ($reverselosses[0] < $sorted[1]) { exit 0 } else { exit 1 }' $testlosses; then
        echo "copying" $networkpath "to" $bestnetworkpath
        cp $networkpath $bestnetworkpath
    fi

    if perl -e 'use warnings; use strict; my @losses = (); while(<>) { chomp($_); next if($_ eq ""); push(@losses, $_) }; if (scalar(@losses) < 10) { exit 1 }; my $anyimprovement = 0; my @reverselosses = reverse(@losses); for (my $i = 0; $i < 9; $i++) { $anyimprovement = $anyimprovement || ($reverselosses[$i] < $reverselosses[9]) }; if ($anyimprovement) { exit 1 } else { exit 0 }' $testlosses; then
        echo "reached a level where it no longer improves"
        exit
    fi
done
