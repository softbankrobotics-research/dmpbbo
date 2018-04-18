message("${LWPR_LIBRARIES}")

add_executable(demoDmp demoDmp.cpp)
target_link_libraries(demoDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmp DESTINATION bin)

add_executable(demoDmpChangeGoal demoDmpChangeGoal.cpp)
target_link_libraries(demoDmpChangeGoal dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpChangeGoal DESTINATION bin)

add_executable(testChangeGoal testChangeGoal.cpp)
target_link_libraries(testChangeGoal dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS testChangeGoal DESTINATION bin)

add_executable(demoDmpContextual demoDmpContextual.cpp)
target_link_libraries(demoDmpContextual dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpContextual DESTINATION bin)

add_executable(demoDmpContextualGoal demoDmpContextualGoal.cpp)
target_link_libraries(demoDmpContextualGoal dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpContextualGoal DESTINATION bin)

add_executable(demoDmpTrainFromTrajectoryFile demoDmpTrainFromTrajectoryFile.cpp)
target_link_libraries(demoDmpTrainFromTrajectoryFile dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpTrainFromTrajectoryFile DESTINATION bin)

add_executable(demoDmpForRobot demoDmpForRobot.cpp)
target_link_libraries(demoDmpForRobot dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpForRobot DESTINATION bin)
