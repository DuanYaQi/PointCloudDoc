# ResFlow



```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
    LogitTransform-1            [-1, 3, 32, 32]               0
    
    
    
#1st-stackediresblocks
    LogitTransform-2            [-1, 3, 32, 32]               0
    
         ActNorm2d-3            [-1, 3, 32, 32]               6
#1st-iresblocks
 InducedNormConv2d-4          [-1, 512, 32, 32]          14,336
             Swish-5          [-1, 512, 32, 32]               1
 InducedNormConv2d-6          [-1, 512, 32, 32]         262,656
             Swish-7          [-1, 512, 32, 32]               1
 InducedNormConv2d-8            [-1, 3, 32, 32]          13,827

         ActNorm2d-9            [-1, 3, 32, 32]               6
#2nd-iresblocks
            Swish-10            [-1, 3, 32, 32]               1
InducedNormConv2d-11          [-1, 512, 32, 32]          14,336
            Swish-12          [-1, 512, 32, 32]               1
InducedNormConv2d-13          [-1, 512, 32, 32]         262,656
            Swish-14          [-1, 512, 32, 32]               1
InducedNormConv2d-15            [-1, 3, 32, 32]          13,827

        ActNorm2d-16            [-1, 3, 32, 32]               6
#3rd-iresblocks
            Swish-17            [-1, 3, 32, 32]               1
InducedNormConv2d-18          [-1, 512, 32, 32]          14,336
            Swish-19          [-1, 512, 32, 32]               1
InducedNormConv2d-20          [-1, 512, 32, 32]         262,656
            Swish-21          [-1, 512, 32, 32]               1
InducedNormConv2d-22            [-1, 3, 32, 32]          13,827

        ActNorm2d-23            [-1, 3, 32, 32]               6
#4th-iresblocks
            Swish-24            [-1, 3, 32, 32]               1
InducedNormConv2d-25          [-1, 512, 32, 32]          14,336
            Swish-26          [-1, 512, 32, 32]               1
InducedNormConv2d-27          [-1, 512, 32, 32]         262,656
            Swish-28          [-1, 512, 32, 32]               1
InducedNormConv2d-29            [-1, 3, 32, 32]          13,827

        ActNorm2d-30            [-1, 3, 32, 32]               6
#5th-iresblocks   
            Swish-31            [-1, 3, 32, 32]               1
InducedNormConv2d-32          [-1, 512, 32, 32]          14,336
            Swish-33          [-1, 512, 32, 32]               1
InducedNormConv2d-34          [-1, 512, 32, 32]         262,656
            Swish-35          [-1, 512, 32, 32]               1
InducedNormConv2d-36            [-1, 3, 32, 32]          13,827
        ActNorm2d-37            [-1, 3, 32, 32]               6
            Swish-38            [-1, 3, 32, 32]               1
InducedNormConv2d-39          [-1, 512, 32, 32]          14,336
            Swish-40          [-1, 512, 32, 32]               1
InducedNormConv2d-41          [-1, 512, 32, 32]         262,656
            Swish-42          [-1, 512, 32, 32]               1
InducedNormConv2d-43            [-1, 3, 32, 32]          13,827
        ActNorm2d-44            [-1, 3, 32, 32]               6
            Swish-45            [-1, 3, 32, 32]               1
InducedNormConv2d-46          [-1, 512, 32, 32]          14,336
            Swish-47          [-1, 512, 32, 32]               1
InducedNormConv2d-48          [-1, 512, 32, 32]         262,656
            Swish-49          [-1, 512, 32, 32]               1
InducedNormConv2d-50            [-1, 3, 32, 32]          13,827
        ActNorm2d-51            [-1, 3, 32, 32]               6
            Swish-52            [-1, 3, 32, 32]               1
InducedNormConv2d-53          [-1, 512, 32, 32]          14,336
            Swish-54          [-1, 512, 32, 32]               1
InducedNormConv2d-55          [-1, 512, 32, 32]         262,656
            Swish-56          [-1, 512, 32, 32]               1
InducedNormConv2d-57            [-1, 3, 32, 32]          13,827
        ActNorm2d-58            [-1, 3, 32, 32]               6
            Swish-59            [-1, 3, 32, 32]               1
InducedNormConv2d-60          [-1, 512, 32, 32]          14,336
            Swish-61          [-1, 512, 32, 32]               1
InducedNormConv2d-62          [-1, 512, 32, 32]         262,656
            Swish-63          [-1, 512, 32, 32]               1
InducedNormConv2d-64            [-1, 3, 32, 32]          13,827
        ActNorm2d-65            [-1, 3, 32, 32]               6
            Swish-66            [-1, 3, 32, 32]               1
InducedNormConv2d-67          [-1, 512, 32, 32]          14,336
            Swish-68          [-1, 512, 32, 32]               1
InducedNormConv2d-69          [-1, 512, 32, 32]         262,656
            Swish-70          [-1, 512, 32, 32]               1
InducedNormConv2d-71            [-1, 3, 32, 32]          13,827
        ActNorm2d-72            [-1, 3, 32, 32]               6
            Swish-73            [-1, 3, 32, 32]               1
InducedNormConv2d-74          [-1, 512, 32, 32]          14,336
            Swish-75          [-1, 512, 32, 32]               1
InducedNormConv2d-76          [-1, 512, 32, 32]         262,656
            Swish-77          [-1, 512, 32, 32]               1
InducedNormConv2d-78            [-1, 3, 32, 32]          13,827
        ActNorm2d-79            [-1, 3, 32, 32]               6
            Swish-80            [-1, 3, 32, 32]               1
InducedNormConv2d-81          [-1, 512, 32, 32]          14,336
            Swish-82          [-1, 512, 32, 32]               1
InducedNormConv2d-83          [-1, 512, 32, 32]         262,656
            Swish-84          [-1, 512, 32, 32]               1
InducedNormConv2d-85            [-1, 3, 32, 32]          13,827
        ActNorm2d-86            [-1, 3, 32, 32]               6
            Swish-87            [-1, 3, 32, 32]               1
InducedNormConv2d-88          [-1, 512, 32, 32]          14,336
            Swish-89          [-1, 512, 32, 32]               1
InducedNormConv2d-90          [-1, 512, 32, 32]         262,656
            Swish-91          [-1, 512, 32, 32]               1
InducedNormConv2d-92            [-1, 3, 32, 32]          13,827
        ActNorm2d-93            [-1, 3, 32, 32]               6
            Swish-94            [-1, 3, 32, 32]               1
InducedNormConv2d-95          [-1, 512, 32, 32]          14,336
            Swish-96          [-1, 512, 32, 32]               1
InducedNormConv2d-97          [-1, 512, 32, 32]         262,656
            Swish-98          [-1, 512, 32, 32]               1
InducedNormConv2d-99            [-1, 3, 32, 32]          13,827
       ActNorm2d-100            [-1, 3, 32, 32]               6
           Swish-101            [-1, 3, 32, 32]               1
InducedNormConv2d-102          [-1, 512, 32, 32]          14,336
           Swish-103          [-1, 512, 32, 32]               1
InducedNormConv2d-104          [-1, 512, 32, 32]         262,656
           Swish-105          [-1, 512, 32, 32]               1
InducedNormConv2d-106            [-1, 3, 32, 32]          13,827
       ActNorm2d-107            [-1, 3, 32, 32]               6
           Swish-108            [-1, 3, 32, 32]               1
InducedNormConv2d-109          [-1, 512, 32, 32]          14,336
           Swish-110          [-1, 512, 32, 32]               1
InducedNormConv2d-111          [-1, 512, 32, 32]         262,656
           Swish-112          [-1, 512, 32, 32]               1
InducedNormConv2d-113            [-1, 3, 32, 32]          13,827
       ActNorm2d-114            [-1, 3, 32, 32]               6
    
    
    
    
    SqueezeLayer-115           [-1, 12, 16, 16]               0
    
    
    
    
           Swish-116           [-1, 12, 16, 16]               1
InducedNormConv2d-117          [-1, 512, 16, 16]          55,808
           Swish-118          [-1, 512, 16, 16]               1
InducedNormConv2d-119          [-1, 512, 16, 16]         262,656
           Swish-120          [-1, 512, 16, 16]               1
InducedNormConv2d-121           [-1, 12, 16, 16]          55,308
       ActNorm2d-122           [-1, 12, 16, 16]              24
           Swish-123           [-1, 12, 16, 16]               1
InducedNormConv2d-124          [-1, 512, 16, 16]          55,808
           Swish-125          [-1, 512, 16, 16]               1
InducedNormConv2d-126          [-1, 512, 16, 16]         262,656
           Swish-127          [-1, 512, 16, 16]               1
InducedNormConv2d-128           [-1, 12, 16, 16]          55,308
       ActNorm2d-129           [-1, 12, 16, 16]              24
           Swish-130           [-1, 12, 16, 16]               1
InducedNormConv2d-131          [-1, 512, 16, 16]          55,808
           Swish-132          [-1, 512, 16, 16]               1
InducedNormConv2d-133          [-1, 512, 16, 16]         262,656
           Swish-134          [-1, 512, 16, 16]               1
InducedNormConv2d-135           [-1, 12, 16, 16]          55,308
       ActNorm2d-136           [-1, 12, 16, 16]              24
           Swish-137           [-1, 12, 16, 16]               1
InducedNormConv2d-138          [-1, 512, 16, 16]          55,808
           Swish-139          [-1, 512, 16, 16]               1
InducedNormConv2d-140          [-1, 512, 16, 16]         262,656
           Swish-141          [-1, 512, 16, 16]               1
InducedNormConv2d-142           [-1, 12, 16, 16]          55,308
       ActNorm2d-143           [-1, 12, 16, 16]              24
           Swish-144           [-1, 12, 16, 16]               1
InducedNormConv2d-145          [-1, 512, 16, 16]          55,808
           Swish-146          [-1, 512, 16, 16]               1
InducedNormConv2d-147          [-1, 512, 16, 16]         262,656
           Swish-148          [-1, 512, 16, 16]               1
InducedNormConv2d-149           [-1, 12, 16, 16]          55,308
       ActNorm2d-150           [-1, 12, 16, 16]              24
           Swish-151           [-1, 12, 16, 16]               1
InducedNormConv2d-152          [-1, 512, 16, 16]          55,808
           Swish-153          [-1, 512, 16, 16]               1
InducedNormConv2d-154          [-1, 512, 16, 16]         262,656
           Swish-155          [-1, 512, 16, 16]               1
InducedNormConv2d-156           [-1, 12, 16, 16]          55,308
       ActNorm2d-157           [-1, 12, 16, 16]              24
           Swish-158           [-1, 12, 16, 16]               1
InducedNormConv2d-159          [-1, 512, 16, 16]          55,808
           Swish-160          [-1, 512, 16, 16]               1
InducedNormConv2d-161          [-1, 512, 16, 16]         262,656
           Swish-162          [-1, 512, 16, 16]               1
InducedNormConv2d-163           [-1, 12, 16, 16]          55,308
       ActNorm2d-164           [-1, 12, 16, 16]              24
           Swish-165           [-1, 12, 16, 16]               1
InducedNormConv2d-166          [-1, 512, 16, 16]          55,808
           Swish-167          [-1, 512, 16, 16]               1
InducedNormConv2d-168          [-1, 512, 16, 16]         262,656
           Swish-169          [-1, 512, 16, 16]               1
InducedNormConv2d-170           [-1, 12, 16, 16]          55,308
       ActNorm2d-171           [-1, 12, 16, 16]              24
           Swish-172           [-1, 12, 16, 16]               1
InducedNormConv2d-173          [-1, 512, 16, 16]          55,808
           Swish-174          [-1, 512, 16, 16]               1
InducedNormConv2d-175          [-1, 512, 16, 16]         262,656
           Swish-176          [-1, 512, 16, 16]               1
InducedNormConv2d-177           [-1, 12, 16, 16]          55,308
       ActNorm2d-178           [-1, 12, 16, 16]              24
           Swish-179           [-1, 12, 16, 16]               1
InducedNormConv2d-180          [-1, 512, 16, 16]          55,808
           Swish-181          [-1, 512, 16, 16]               1
InducedNormConv2d-182          [-1, 512, 16, 16]         262,656
           Swish-183          [-1, 512, 16, 16]               1
InducedNormConv2d-184           [-1, 12, 16, 16]          55,308
       ActNorm2d-185           [-1, 12, 16, 16]              24
           Swish-186           [-1, 12, 16, 16]               1
InducedNormConv2d-187          [-1, 512, 16, 16]          55,808
           Swish-188          [-1, 512, 16, 16]               1
InducedNormConv2d-189          [-1, 512, 16, 16]         262,656
           Swish-190          [-1, 512, 16, 16]               1
InducedNormConv2d-191           [-1, 12, 16, 16]          55,308
       ActNorm2d-192           [-1, 12, 16, 16]              24
           Swish-193           [-1, 12, 16, 16]               1
InducedNormConv2d-194          [-1, 512, 16, 16]          55,808
           Swish-195          [-1, 512, 16, 16]               1
InducedNormConv2d-196          [-1, 512, 16, 16]         262,656
           Swish-197          [-1, 512, 16, 16]               1
InducedNormConv2d-198           [-1, 12, 16, 16]          55,308
       ActNorm2d-199           [-1, 12, 16, 16]              24
           Swish-200           [-1, 12, 16, 16]               1
InducedNormConv2d-201          [-1, 512, 16, 16]          55,808
           Swish-202          [-1, 512, 16, 16]               1
InducedNormConv2d-203          [-1, 512, 16, 16]         262,656
           Swish-204          [-1, 512, 16, 16]               1
InducedNormConv2d-205           [-1, 12, 16, 16]          55,308
       ActNorm2d-206           [-1, 12, 16, 16]              24
           Swish-207           [-1, 12, 16, 16]               1
InducedNormConv2d-208          [-1, 512, 16, 16]          55,808
           Swish-209          [-1, 512, 16, 16]               1
InducedNormConv2d-210          [-1, 512, 16, 16]         262,656
           Swish-211          [-1, 512, 16, 16]               1
InducedNormConv2d-212           [-1, 12, 16, 16]          55,308
       ActNorm2d-213           [-1, 12, 16, 16]              24
           Swish-214           [-1, 12, 16, 16]               1
InducedNormConv2d-215          [-1, 512, 16, 16]          55,808
           Swish-216          [-1, 512, 16, 16]               1
InducedNormConv2d-217          [-1, 512, 16, 16]         262,656
           Swish-218          [-1, 512, 16, 16]               1
InducedNormConv2d-219           [-1, 12, 16, 16]          55,308
       ActNorm2d-220           [-1, 12, 16, 16]              24
           Swish-221           [-1, 12, 16, 16]               1
InducedNormConv2d-222          [-1, 512, 16, 16]          55,808
           Swish-223          [-1, 512, 16, 16]               1
InducedNormConv2d-224          [-1, 512, 16, 16]         262,656
           Swish-225          [-1, 512, 16, 16]               1
InducedNormConv2d-226           [-1, 12, 16, 16]          55,308
       ActNorm2d-227           [-1, 12, 16, 16]              24
    
    
    
    SqueezeLayer-228             [-1, 48, 8, 8]               0
    
    
    
           Swish-229             [-1, 48, 8, 8]               1
InducedNormConv2d-230            [-1, 512, 8, 8]         221,696
           Swish-231            [-1, 512, 8, 8]               1
InducedNormConv2d-232            [-1, 512, 8, 8]         262,656
           Swish-233            [-1, 512, 8, 8]               1
InducedNormConv2d-234             [-1, 48, 8, 8]         221,232
       ActNorm2d-235             [-1, 48, 8, 8]              96
           Swish-236             [-1, 48, 8, 8]               1
InducedNormConv2d-237            [-1, 512, 8, 8]         221,696
           Swish-238            [-1, 512, 8, 8]               1
InducedNormConv2d-239            [-1, 512, 8, 8]         262,656
           Swish-240            [-1, 512, 8, 8]               1
InducedNormConv2d-241             [-1, 48, 8, 8]         221,232
       ActNorm2d-242             [-1, 48, 8, 8]              96
           Swish-243             [-1, 48, 8, 8]               1
InducedNormConv2d-244            [-1, 512, 8, 8]         221,696
           Swish-245            [-1, 512, 8, 8]               1
InducedNormConv2d-246            [-1, 512, 8, 8]         262,656
           Swish-247            [-1, 512, 8, 8]               1
InducedNormConv2d-248             [-1, 48, 8, 8]         221,232
       ActNorm2d-249             [-1, 48, 8, 8]              96
           Swish-250             [-1, 48, 8, 8]               1
InducedNormConv2d-251            [-1, 512, 8, 8]         221,696
           Swish-252            [-1, 512, 8, 8]               1
InducedNormConv2d-253            [-1, 512, 8, 8]         262,656
           Swish-254            [-1, 512, 8, 8]               1
InducedNormConv2d-255             [-1, 48, 8, 8]         221,232
       ActNorm2d-256             [-1, 48, 8, 8]              96
           Swish-257             [-1, 48, 8, 8]               1
InducedNormConv2d-258            [-1, 512, 8, 8]         221,696
           Swish-259            [-1, 512, 8, 8]               1
InducedNormConv2d-260            [-1, 512, 8, 8]         262,656
           Swish-261            [-1, 512, 8, 8]               1
InducedNormConv2d-262             [-1, 48, 8, 8]         221,232
       ActNorm2d-263             [-1, 48, 8, 8]              96
           Swish-264             [-1, 48, 8, 8]               1
InducedNormConv2d-265            [-1, 512, 8, 8]         221,696
           Swish-266            [-1, 512, 8, 8]               1
InducedNormConv2d-267            [-1, 512, 8, 8]         262,656
           Swish-268            [-1, 512, 8, 8]               1
InducedNormConv2d-269             [-1, 48, 8, 8]         221,232
       ActNorm2d-270             [-1, 48, 8, 8]              96
           Swish-271             [-1, 48, 8, 8]               1
InducedNormConv2d-272            [-1, 512, 8, 8]         221,696
           Swish-273            [-1, 512, 8, 8]               1
InducedNormConv2d-274            [-1, 512, 8, 8]         262,656
           Swish-275            [-1, 512, 8, 8]               1
InducedNormConv2d-276             [-1, 48, 8, 8]         221,232
       ActNorm2d-277             [-1, 48, 8, 8]              96
           Swish-278             [-1, 48, 8, 8]               1
InducedNormConv2d-279            [-1, 512, 8, 8]         221,696
           Swish-280            [-1, 512, 8, 8]               1
InducedNormConv2d-281            [-1, 512, 8, 8]         262,656
           Swish-282            [-1, 512, 8, 8]               1
InducedNormConv2d-283             [-1, 48, 8, 8]         221,232
       ActNorm2d-284             [-1, 48, 8, 8]              96
           Swish-285             [-1, 48, 8, 8]               1
InducedNormConv2d-286            [-1, 512, 8, 8]         221,696
           Swish-287            [-1, 512, 8, 8]               1
InducedNormConv2d-288            [-1, 512, 8, 8]         262,656
           Swish-289            [-1, 512, 8, 8]               1
InducedNormConv2d-290             [-1, 48, 8, 8]         221,232
       ActNorm2d-291             [-1, 48, 8, 8]              96
           Swish-292             [-1, 48, 8, 8]               1
InducedNormConv2d-293            [-1, 512, 8, 8]         221,696
           Swish-294            [-1, 512, 8, 8]               1
InducedNormConv2d-295            [-1, 512, 8, 8]         262,656
           Swish-296            [-1, 512, 8, 8]               1
InducedNormConv2d-297             [-1, 48, 8, 8]         221,232
       ActNorm2d-298             [-1, 48, 8, 8]              96
           Swish-299             [-1, 48, 8, 8]               1
InducedNormConv2d-300            [-1, 512, 8, 8]         221,696
           Swish-301            [-1, 512, 8, 8]               1
InducedNormConv2d-302            [-1, 512, 8, 8]         262,656
           Swish-303            [-1, 512, 8, 8]               1
InducedNormConv2d-304             [-1, 48, 8, 8]         221,232
       ActNorm2d-305             [-1, 48, 8, 8]              96
           Swish-306             [-1, 48, 8, 8]               1
InducedNormConv2d-307            [-1, 512, 8, 8]         221,696
           Swish-308            [-1, 512, 8, 8]               1
InducedNormConv2d-309            [-1, 512, 8, 8]         262,656
           Swish-310            [-1, 512, 8, 8]               1
InducedNormConv2d-311             [-1, 48, 8, 8]         221,232
       ActNorm2d-312             [-1, 48, 8, 8]              96
           Swish-313             [-1, 48, 8, 8]               1
InducedNormConv2d-314            [-1, 512, 8, 8]         221,696
           Swish-315            [-1, 512, 8, 8]               1
InducedNormConv2d-316            [-1, 512, 8, 8]         262,656
           Swish-317            [-1, 512, 8, 8]               1
InducedNormConv2d-318             [-1, 48, 8, 8]         221,232
       ActNorm2d-319             [-1, 48, 8, 8]              96
           Swish-320             [-1, 48, 8, 8]               1
InducedNormConv2d-321            [-1, 512, 8, 8]         221,696
           Swish-322            [-1, 512, 8, 8]               1
InducedNormConv2d-323            [-1, 512, 8, 8]         262,656
           Swish-324            [-1, 512, 8, 8]               1
InducedNormConv2d-325             [-1, 48, 8, 8]         221,232
       ActNorm2d-326             [-1, 48, 8, 8]              96
           Swish-327             [-1, 48, 8, 8]               1
InducedNormConv2d-328            [-1, 512, 8, 8]         221,696
           Swish-329            [-1, 512, 8, 8]               1
InducedNormConv2d-330            [-1, 512, 8, 8]         262,656
           Swish-331            [-1, 512, 8, 8]               1
InducedNormConv2d-332             [-1, 48, 8, 8]         221,232
       ActNorm2d-333             [-1, 48, 8, 8]              96
           Swish-334             [-1, 48, 8, 8]               1
InducedNormConv2d-335            [-1, 512, 8, 8]         221,696
           Swish-336            [-1, 512, 8, 8]               1
InducedNormConv2d-337            [-1, 512, 8, 8]         262,656
           Swish-338            [-1, 512, 8, 8]               1
InducedNormConv2d-339             [-1, 48, 8, 8]         221,232
       ActNorm2d-340             [-1, 48, 8, 8]              96
           Swish-341                 [-1, 3072]               1
InducedNormLinear-342                  [-1, 128]         393,344
           Swish-343                  [-1, 128]               1
InducedNormLinear-344                  [-1, 128]          16,512
           Swish-345                  [-1, 128]               1
InducedNormLinear-346                 [-1, 3072]         396,288
       ActNorm1d-347                 [-1, 3072]           6,144
           Swish-348                 [-1, 3072]               1
InducedNormLinear-349                  [-1, 128]         393,344
           Swish-350                  [-1, 128]               1
InducedNormLinear-351                  [-1, 128]          16,512
           Swish-352                  [-1, 128]               1
InducedNormLinear-353                 [-1, 3072]         396,288
       ActNorm1d-354                 [-1, 3072]           6,144
           Swish-355                 [-1, 3072]               1
InducedNormLinear-356                  [-1, 128]         393,344
           Swish-357                  [-1, 128]               1
InducedNormLinear-358                  [-1, 128]          16,512
           Swish-359                  [-1, 128]               1
InducedNormLinear-360                 [-1, 3072]         396,288
       ActNorm1d-361                 [-1, 3072]           6,144
           Swish-362                 [-1, 3072]               1
InducedNormLinear-363                  [-1, 128]         393,344
           Swish-364                  [-1, 128]               1
InducedNormLinear-365                  [-1, 128]          16,512
           Swish-366                  [-1, 128]               1
InducedNormLinear-367                 [-1, 3072]         396,288
       ActNorm1d-368                 [-1, 3072]           6,144
================================================================
Total params: 25,174,129
Trainable params: 25,174,129
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.011719
Forward/backward pass size (MB): 339.765625
Params size (MB): 96.031681
Estimated Total Size (MB): 435.809025
----------------------------------------------------------------

```

