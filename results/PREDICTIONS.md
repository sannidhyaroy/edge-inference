# Predictions

Written before any of these were measured, so that the model-menu measurements
test the claim in `docs/DESIGN.md` rather than searching for results that fit
it. The commit that adds this file is the timestamp. Each prediction will be
marked held or failed next to the measured value, and a failed one is
reported, not dropped.

Numbers are guesses with a range, not targets.

---

## Hedging on the early-exit table

1. Over the 16 device-speed and deadline cells, with three seeds, hedging
   without any knowledge of the network scores at least the fair target
   (rung 3) in 12 or more cells, and is never more than 0.5 points below it.
2. The lookup rule over the family that includes hedged plans reaches rung 3h
   within 0.3 points in every cell.
3. Rung 3h is above rung 3 by more than one point only in cells where some
   frames on the good network miss the deadline when sent.
4. Hedging keeps the device busy at least 1.8 times as long as the lookup
   rule in the cells where the lookup rule sends; cancellation recovers more
   than half of that difference.

## The device menu (ImageNetV2, float32, 224 px)

5. Top-1 accuracy: ResNet-18 55 to 60%, MobileNetV3-Large 61 to 66%,
   ResNet-50 (V2 weights) 67 to 72%, ConvNeXt-Tiny 69 to 74%.
6. Static INT8 costs MobileNetV3-Large more than 2 points, MobileNetV4 (conv)
   less than 1.5, ResNet-18 less than 1.
7. Going from 224 to 160 px without fine-tuning costs ResNet-18 3 to 8 points,
   and time falls to 45 to 65% of the 224 px time.
8. On the slowed laptop, time does not scale by one constant: the
   slowed-to-normal ratio of the MobileNets differs from ResNet-18's by more
   than 10%.
9. INT8 lowers energy per inference by at least 30% for ResNet-18.

## Servers and links

10. ConvNeXt-Tiny on the T4 takes less than a third of its time on the second
    laptop's CPU at 6 threads.
11. Median round trip: crowded 2.4 GHz Wi-Fi 5 to 30 ms with a 95th
    percentile above 100 ms; a 6 GHz phone hotspot under 10 ms; a direct
    cable under 1 ms; the remote GPU 150 to 300 ms.

## Deciding on recorded links

12. On the cable and the hotspot, a fixed option is within the paired interval
    of the lookup rule at every deadline: no adaptation is needed.
13. On the crowded Wi-Fi at a 100 ms deadline, adapting beats the best fixed
    option by more than the paired interval.
14. The lookup rule is within 0.5 points of rung 3 on every recorded link.
15. For record with H = 1000 ms, hedging with the remote GPU scores within one
    point of that server's own accuracy.
16. Live, the advisor's recommended plan scores within its paired interval of
    what the simulator predicted for the same recording.
