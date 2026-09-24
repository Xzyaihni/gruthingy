#!/bin/sh

set -euo pipefail
IFS=$'\n\t'

if [ $# -lt 4 ]; then
    echo "usage:" $0 "<output-path>" "<test-input-path>" "<try-n-runs>" "<..command..>"
    exit
fi

networkpath=$1
bestnetworkpath="${1}.best"
echo "saving best in" $bestnetworkpath

testlosses="${1}.test_losses"
echo "saving test losses in" $testlosses

testinput=$2
echo "loading test inputs from" $testinput

trynruns=$3
echo "will try" $trynruns "runs of non decreasing validation loss before giving up"

shift 3

while true
do
    $@

    wait

    # even though this MIGHT finish after the next network trained, it wont do that on any reasonable amount of iterations and a reasonable test file
    {
        $1 --mode test --path $networkpath --input $testinput --batch 5 | grep --only-matching '[\.0-9]*' >> $testlosses

        perlheader="use warnings; use strict; my \$checks = $trynruns;"

        if perl -e $perlheader'use List::Util qw( min ); my @losses = (); while(<>) { chomp($_); next if($_ eq ""); push(@losses, $_) }; if (scalar(@losses) < 2) { exit 0 }; my @reverselosses = (reverse(@losses))[0..(min($checks - 1, scalar(@losses) - 1))]; my @sorted = sort {$a <=> $b} @reverselosses; if ($reverselosses[0] < $sorted[1]) { exit 0 } else { exit 1 }' $testlosses; then
            echo "copying" $networkpath "to" $bestnetworkpath
            cp $networkpath $bestnetworkpath
        fi

        if perl -e $perlheader'my @losses = (); while(<>) { chomp($_); next if($_ eq ""); push(@losses, $_) }; if (scalar(@losses) < $checks) { exit 1 }; my $anyimprovement = 0; my @reverselosses = reverse(@losses); for (my $i = 0; $i < ($checks - 1); $i++) { $anyimprovement = $anyimprovement || ($reverselosses[$i] < $reverselosses[($checks - 1)]) }; if ($anyimprovement) { exit 1 } else { exit 0 }' $testlosses; then
            echo "reached a level where it no longer improves"
            exit
        fi
    } &
done
