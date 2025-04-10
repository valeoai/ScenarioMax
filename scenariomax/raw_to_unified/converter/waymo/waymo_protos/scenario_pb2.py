# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scenariomax/raw_to_unified/converter/waymo/waymo_protos/scenario.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from scenariomax.raw_to_unified.converter.waymo.waymo_protos import compressed_lidar_pb2 as scenariomax_dot_raw__to__unified_dot_converter_dot_waymo_dot_waymo__protos_dot_compressed__lidar__pb2
from scenariomax.raw_to_unified.converter.waymo.waymo_protos import map_pb2 as scenariomax_dot_raw__to__unified_dot_converter_dot_waymo_dot_waymo__protos_dot_map__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='scenariomax/raw_to_unified/converter/waymo/waymo_protos/scenario.proto',
  package='waymo.open_dataset',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\nFscenariomax/raw_to_unified/converter/waymo/waymo_protos/scenario.proto\x12\x12waymo.open_dataset\x1aNscenariomax/raw_to_unified/converter/waymo/waymo_protos/compressed_lidar.proto\x1a\x41scenariomax/raw_to_unified/converter/waymo/waymo_protos/map.proto\"\xba\x01\n\x0bObjectState\x12\x10\n\x08\x63\x65nter_x\x18\x02 \x01(\x01\x12\x10\n\x08\x63\x65nter_y\x18\x03 \x01(\x01\x12\x10\n\x08\x63\x65nter_z\x18\x04 \x01(\x01\x12\x0e\n\x06length\x18\x05 \x01(\x02\x12\r\n\x05width\x18\x06 \x01(\x02\x12\x0e\n\x06height\x18\x07 \x01(\x02\x12\x0f\n\x07heading\x18\x08 \x01(\x02\x12\x12\n\nvelocity_x\x18\t \x01(\x02\x12\x12\n\nvelocity_y\x18\n \x01(\x02\x12\r\n\x05valid\x18\x0b \x01(\x08\"\xe6\x01\n\x05Track\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x39\n\x0bobject_type\x18\x02 \x01(\x0e\x32$.waymo.open_dataset.Track.ObjectType\x12/\n\x06states\x18\x03 \x03(\x0b\x32\x1f.waymo.open_dataset.ObjectState\"e\n\nObjectType\x12\x0e\n\nTYPE_UNSET\x10\x00\x12\x10\n\x0cTYPE_VEHICLE\x10\x01\x12\x13\n\x0fTYPE_PEDESTRIAN\x10\x02\x12\x10\n\x0cTYPE_CYCLIST\x10\x03\x12\x0e\n\nTYPE_OTHER\x10\x04\"R\n\x0f\x44ynamicMapState\x12?\n\x0blane_states\x18\x01 \x03(\x0b\x32*.waymo.open_dataset.TrafficSignalLaneState\"\xac\x01\n\x12RequiredPrediction\x12\x13\n\x0btrack_index\x18\x01 \x01(\x05\x12J\n\ndifficulty\x18\x02 \x01(\x0e\x32\x36.waymo.open_dataset.RequiredPrediction.DifficultyLevel\"5\n\x0f\x44ifficultyLevel\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07LEVEL_1\x10\x01\x12\x0b\n\x07LEVEL_2\x10\x02\"\xcb\x03\n\x08Scenario\x12\x13\n\x0bscenario_id\x18\x05 \x01(\t\x12\x1a\n\x12timestamps_seconds\x18\x01 \x03(\x01\x12\x1a\n\x12\x63urrent_time_index\x18\n \x01(\x05\x12)\n\x06tracks\x18\x02 \x03(\x0b\x32\x19.waymo.open_dataset.Track\x12?\n\x12\x64ynamic_map_states\x18\x07 \x03(\x0b\x32#.waymo.open_dataset.DynamicMapState\x12\x34\n\x0cmap_features\x18\x08 \x03(\x0b\x32\x1e.waymo.open_dataset.MapFeature\x12\x17\n\x0fsdc_track_index\x18\x06 \x01(\x05\x12\x1b\n\x13objects_of_interest\x18\x04 \x03(\x05\x12\x41\n\x11tracks_to_predict\x18\x0b \x03(\x0b\x32&.waymo.open_dataset.RequiredPrediction\x12Q\n\x1b\x63ompressed_frame_laser_data\x18\x0c \x03(\x0b\x32,.waymo.open_dataset.CompressedFrameLaserDataJ\x04\x08\t\x10\n')
  ,
  dependencies=[scenariomax_dot_raw__to__unified_dot_converter_dot_waymo_dot_waymo__protos_dot_compressed__lidar__pb2.DESCRIPTOR,scenariomax_dot_raw__to__unified_dot_converter_dot_waymo_dot_waymo__protos_dot_map__pb2.DESCRIPTOR,])



_TRACK_OBJECTTYPE = _descriptor.EnumDescriptor(
  name='ObjectType',
  full_name='waymo.open_dataset.Track.ObjectType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TYPE_UNSET', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_VEHICLE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_PEDESTRIAN', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_CYCLIST', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_OTHER', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=560,
  serialized_end=661,
)
_sym_db.RegisterEnumDescriptor(_TRACK_OBJECTTYPE)

_REQUIREDPREDICTION_DIFFICULTYLEVEL = _descriptor.EnumDescriptor(
  name='DifficultyLevel',
  full_name='waymo.open_dataset.RequiredPrediction.DifficultyLevel',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LEVEL_1', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LEVEL_2', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=867,
  serialized_end=920,
)
_sym_db.RegisterEnumDescriptor(_REQUIREDPREDICTION_DIFFICULTYLEVEL)


_OBJECTSTATE = _descriptor.Descriptor(
  name='ObjectState',
  full_name='waymo.open_dataset.ObjectState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='center_x', full_name='waymo.open_dataset.ObjectState.center_x', index=0,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='center_y', full_name='waymo.open_dataset.ObjectState.center_y', index=1,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='center_z', full_name='waymo.open_dataset.ObjectState.center_z', index=2,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='length', full_name='waymo.open_dataset.ObjectState.length', index=3,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='waymo.open_dataset.ObjectState.width', index=4,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='waymo.open_dataset.ObjectState.height', index=5,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='heading', full_name='waymo.open_dataset.ObjectState.heading', index=6,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='velocity_x', full_name='waymo.open_dataset.ObjectState.velocity_x', index=7,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='velocity_y', full_name='waymo.open_dataset.ObjectState.velocity_y', index=8,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='valid', full_name='waymo.open_dataset.ObjectState.valid', index=9,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=242,
  serialized_end=428,
)


_TRACK = _descriptor.Descriptor(
  name='Track',
  full_name='waymo.open_dataset.Track',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='waymo.open_dataset.Track.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object_type', full_name='waymo.open_dataset.Track.object_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='states', full_name='waymo.open_dataset.Track.states', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TRACK_OBJECTTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=431,
  serialized_end=661,
)


_DYNAMICMAPSTATE = _descriptor.Descriptor(
  name='DynamicMapState',
  full_name='waymo.open_dataset.DynamicMapState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lane_states', full_name='waymo.open_dataset.DynamicMapState.lane_states', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=663,
  serialized_end=745,
)


_REQUIREDPREDICTION = _descriptor.Descriptor(
  name='RequiredPrediction',
  full_name='waymo.open_dataset.RequiredPrediction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='track_index', full_name='waymo.open_dataset.RequiredPrediction.track_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='difficulty', full_name='waymo.open_dataset.RequiredPrediction.difficulty', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _REQUIREDPREDICTION_DIFFICULTYLEVEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=748,
  serialized_end=920,
)


_SCENARIO = _descriptor.Descriptor(
  name='Scenario',
  full_name='waymo.open_dataset.Scenario',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scenario_id', full_name='waymo.open_dataset.Scenario.scenario_id', index=0,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='timestamps_seconds', full_name='waymo.open_dataset.Scenario.timestamps_seconds', index=1,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='current_time_index', full_name='waymo.open_dataset.Scenario.current_time_index', index=2,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tracks', full_name='waymo.open_dataset.Scenario.tracks', index=3,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dynamic_map_states', full_name='waymo.open_dataset.Scenario.dynamic_map_states', index=4,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='map_features', full_name='waymo.open_dataset.Scenario.map_features', index=5,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sdc_track_index', full_name='waymo.open_dataset.Scenario.sdc_track_index', index=6,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='objects_of_interest', full_name='waymo.open_dataset.Scenario.objects_of_interest', index=7,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tracks_to_predict', full_name='waymo.open_dataset.Scenario.tracks_to_predict', index=8,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='compressed_frame_laser_data', full_name='waymo.open_dataset.Scenario.compressed_frame_laser_data', index=9,
      number=12, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=923,
  serialized_end=1382,
)

_TRACK.fields_by_name['object_type'].enum_type = _TRACK_OBJECTTYPE
_TRACK.fields_by_name['states'].message_type = _OBJECTSTATE
_TRACK_OBJECTTYPE.containing_type = _TRACK
_DYNAMICMAPSTATE.fields_by_name['lane_states'].message_type = scenariomax_dot_raw__to__unified_dot_converter_dot_waymo_dot_waymo__protos_dot_map__pb2._TRAFFICSIGNALLANESTATE
_REQUIREDPREDICTION.fields_by_name['difficulty'].enum_type = _REQUIREDPREDICTION_DIFFICULTYLEVEL
_REQUIREDPREDICTION_DIFFICULTYLEVEL.containing_type = _REQUIREDPREDICTION
_SCENARIO.fields_by_name['tracks'].message_type = _TRACK
_SCENARIO.fields_by_name['dynamic_map_states'].message_type = _DYNAMICMAPSTATE
_SCENARIO.fields_by_name['map_features'].message_type = scenariomax_dot_raw__to__unified_dot_converter_dot_waymo_dot_waymo__protos_dot_map__pb2._MAPFEATURE
_SCENARIO.fields_by_name['tracks_to_predict'].message_type = _REQUIREDPREDICTION
_SCENARIO.fields_by_name['compressed_frame_laser_data'].message_type = scenariomax_dot_raw__to__unified_dot_converter_dot_waymo_dot_waymo__protos_dot_compressed__lidar__pb2._COMPRESSEDFRAMELASERDATA
DESCRIPTOR.message_types_by_name['ObjectState'] = _OBJECTSTATE
DESCRIPTOR.message_types_by_name['Track'] = _TRACK
DESCRIPTOR.message_types_by_name['DynamicMapState'] = _DYNAMICMAPSTATE
DESCRIPTOR.message_types_by_name['RequiredPrediction'] = _REQUIREDPREDICTION
DESCRIPTOR.message_types_by_name['Scenario'] = _SCENARIO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ObjectState = _reflection.GeneratedProtocolMessageType('ObjectState', (_message.Message,), dict(
  DESCRIPTOR = _OBJECTSTATE,
  __module__ = 'scenariomax.raw_to_unified.converter.waymo.waymo_protos.scenario_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.ObjectState)
  ))
_sym_db.RegisterMessage(ObjectState)

Track = _reflection.GeneratedProtocolMessageType('Track', (_message.Message,), dict(
  DESCRIPTOR = _TRACK,
  __module__ = 'scenariomax.raw_to_unified.converter.waymo.waymo_protos.scenario_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Track)
  ))
_sym_db.RegisterMessage(Track)

DynamicMapState = _reflection.GeneratedProtocolMessageType('DynamicMapState', (_message.Message,), dict(
  DESCRIPTOR = _DYNAMICMAPSTATE,
  __module__ = 'scenariomax.raw_to_unified.converter.waymo.waymo_protos.scenario_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.DynamicMapState)
  ))
_sym_db.RegisterMessage(DynamicMapState)

RequiredPrediction = _reflection.GeneratedProtocolMessageType('RequiredPrediction', (_message.Message,), dict(
  DESCRIPTOR = _REQUIREDPREDICTION,
  __module__ = 'scenariomax.raw_to_unified.converter.waymo.waymo_protos.scenario_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.RequiredPrediction)
  ))
_sym_db.RegisterMessage(RequiredPrediction)

Scenario = _reflection.GeneratedProtocolMessageType('Scenario', (_message.Message,), dict(
  DESCRIPTOR = _SCENARIO,
  __module__ = 'scenariomax.raw_to_unified.converter.waymo.waymo_protos.scenario_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Scenario)
  ))
_sym_db.RegisterMessage(Scenario)


# @@protoc_insertion_point(module_scope)
