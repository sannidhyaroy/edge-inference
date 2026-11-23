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

   **Failed on the count, held on the margin.** Hedging reaches rung 3 in 9
   cells of 16, beating it by up to 2.12 points where deciding pays, and is
   never more than 0.11 points below it on average over seeds (0.19 on the
   worst seed). It needs no knowledge of the network to get there.
   (`decide_sweep_..._hedge.csv`)
2. The lookup rule over the family that includes hedged plans reaches rung 3h
   within 0.3 points in every cell.

   **Held** on cell means: at most 0.17 points apart. One seed in one cell
   was 0.33 points apart.
3. Rung 3h is above rung 3 by more than one point only in cells where some
   frames on the good network miss the deadline when sent.

   **Held.** Rung 3h is more than a point above rung 3 only at 4x with 20 ms
   (2.20) and 8x with 30 ms (1.19); everywhere else the gap is at most 0.68.
4. Hedging keeps the device busy at least 1.8 times as long as the lookup
   rule in the cells where the lookup rule sends; cancellation recovers more
   than half of that difference.

   **Failed.** In 8 of the 13 cells where the lookup rule sends, hedging keeps
   the device busy 1.98 to 2.18 times as long; in 4 it picks a cheaper local
   plan and is busy for less. Cancelling at the next exit recovers at most
   52% of the gap, and none of it where deciding pays: there the local plan is
   a single stage, the server's answer arrives inside it, and the next exit is
   the end of the plan. Stopping the moment the answer arrives does better: it
   recovers 46% at 8x with 20 and 30 ms, 72% at 16x, and more than half in 6
   of the 13 cells, with the same answers. It assumes aborting a running graph
   costs nothing, which the live runtime will measure.
   (`decide_sweep_..._hedge-cancel.csv`, `decide_sweep_..._hedge-cancel-now.csv`)

## The device menu (ImageNetV2, float32, 224 px)

5. Top-1 accuracy: ResNet-18 55 to 60%, MobileNetV3-Large 61 to 66%,
   ResNet-50 (V2 weights) 67 to 72%, ConvNeXt-Tiny 69 to 74%.

   **Held** for all four, on the 5,000 test images: ResNet-18 57.14%,
   MobileNetV3-Large 61.18%, ResNet-50 69.68%, ConvNeXt-Tiny 70.54%.
   (`menu_answers_*.csv.gz`)
6. Static INT8 costs MobileNetV3-Large more than 2 points, MobileNetV4 (conv)
   less than 1.5, ResNet-18 less than 1.

   **Failed** with the default recipe (unsigned activations, signed weights,
   ranges from minimum to maximum): it cost MobileNetV3-Large 23.86 points,
   MobileNetV4-small 24.64, MobileNetV4-medium 3.86 and ResNet-18 2.90. Part
   of the collapse is this processor's 8-bit arithmetic without VNNI, which
   saturates a 16-bit sum; the rest is ranges stretched by outliers. With 7-bit
   weights, percentile ranges and the first and last layers kept in float
   (`--int8-recipe no-vnni`) the costs are 3.02, 7.76, 0.98 and 0.24: the
   prediction then holds for MobileNetV4-medium and ResNet-18 and still fails
   for MobileNetV4-small.
7. Going from 224 to 160 px without fine-tuning costs ResNet-18 3 to 8 points,
   and time falls to 45 to 65% of the 224 px time.

   **Held**: 7.72 points (57.14% to 49.42%), and 53.7% of the time in float32
   (53.2% in INT8), at full speed with two threads.
8. On the slowed laptop, time does not scale by one constant: the
   slowed-to-normal ratio of the MobileNets differs from ResNet-18's by more
   than 10%.

   **Failed.** Per model, the median ratio over its variants is 1.89 to 1.99
   at 2.0 GHz and 2.53 to 2.77 at 1.4 GHz; the MobileNets differ from
   ResNet-18 by 0.5 to 5.2% and by 2.1 to 8.8%. Within a model the ratio does
   vary with the input size, from 1.70 to 2.05 at 2.0 GHz, smallest for the
   smallest inputs, so one constant factor is a fair first model of a slower
   device but not an exact one. (`menu_device_laptop-*.csv.gz`)
9. INT8 lowers energy per inference by at least 30% for ResNet-18.

   **Held at two of three speeds.** Package energy per inference at 224 px
   falls by 32.8% at full speed (446 to 300 mJ) and 32.2% at 2.0 GHz (159 to
   108 mJ), but by 25.6% at 1.4 GHz (180 to 134 mJ, with intervals of about
   ±20 mJ there). (`menu_energy_laptop-*.csv`)

## Servers and links

10. ConvNeXt-Tiny on the T4 takes less than a third of its time on the second
    laptop's CPU at 6 threads.

    **Held:** 9.2 ms per photo at the median on the T4 (decoding and preparing
    3.3 ms, running 5.9 ms), against 54 ms on the second laptop, a sixth.
    (`menu_server_cloud-t4_*`, `menu_server_edge-windows_*`)
11. Median round trip: crowded 2.4 GHz Wi-Fi 5 to 30 ms with a 95th
    percentile above 100 ms; a 6 GHz phone hotspot under 10 ms; a direct
    cable under 1 ms; the remote GPU 150 to 300 ms.

    **Mixed.** The remote GPU, in Oregon: 279 and 292 ms, held. The crowded
    Wi-Fi: 95th percentile 228 to 358 ms, held, but a median of 44 to 49 ms as
    frames meet it (21 ms per completed exchange), failed. The hotspot ran at
    5 GHz, not 6, since the second laptop's Wi-Fi could not use 6 GHz: 10.2 to
    10.5 ms, failed narrowly. The cable: 1.6 to 1.8 ms, failed. (`link_*.csv`)

## Deciding on recorded links

12. On the cable and the hotspot, a fixed option is within the paired interval
    of the lookup rule at every deadline: no adaptation is needed.

    **Failed at 100 and 200 ms, held at 50 ms.** Against the edge server at
    its live speed, with the device at any of three speeds, the lookup rule
    beats the best fixed option (even one picked with hindsight) by 1.26 to
    1.88 points with 95% paired intervals above zero in 11 of the 12 settings
    at 100 and 200 ms; the twelfth is the hotspot at 1.4 GHz and 100 ms. It
    does so with a cascade: answer on the device, and send the photo only
    when unsure. At 50 ms the server cannot answer in time and the fixed
    option is the advice. (`menu_scoreboard_laptop-*.csv`,
    `menu_advice_edge-windows.csv`)
13. On the crowded Wi-Fi at a 100 ms deadline, adapting beats the best fixed
    option by more than the paired interval.

    **Failed.** On the shared Wi-Fi no rule beats the best fixed option at any
    deadline or device speed: sending rarely arrives in time there, so the
    device's own best model is the answer. Adapting paid on the good links,
    the opposite of what was predicted.
14. The lookup rule is within 0.5 points of rung 3 on every recorded link.

    **Held in 25 of 27 settings.** The exceptions: the hotspot at 1.4 GHz and
    100 ms, 1.38 points behind (interval 0.78 to 2.00), and the cable at full
    speed and 50 ms, 0.60 behind (interval -0.06 to 1.24).
15. For record with H = 1000 ms, hedging with the remote GPU scores within one
    point of that server's own accuracy.

    **Held.** The hedge with ConvNeXt V2-Base on the T4 records 73.88% on both
    cloud links, at every device speed and deadline: the server's own accuracy
    on the photos, since every answer arrives within the second. Live, on 200
    frames over the link to Oregon, it recorded 75.0%, which the simulator
    also expected for those images. (`menu_scoreboard_*_cloud-t4-oregon_*`,
    `menu_live_check_laptop-asis-t2_cloud-t4.csv`)
16. Live, the advisor's recommended plan scores within its paired interval of
    what the simulator predicted for the same recording.

    **Not tested as stated.** The live runs carried three fixed plans (local,
    send, hedge) chosen before the advisor existed, not the advisor's cascade.
    For those plans the simulator matched every answer live and came within
    1.6 points of every score on the cable and the hotspot; on the shared
    Wi-Fi it overestimated sending (6.6% against 0.5%), since the live run met
    a busier network than the recording. (`menu_live_check_*.csv`)
