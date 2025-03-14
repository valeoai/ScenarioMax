# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scenariomax/raw_to_unified/converter/waymo/waymo_protos/vector.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='scenariomax/raw_to_unified/converter/waymo/waymo_protos/vector.proto',
  package='waymo.open_dataset',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\nDscenariomax/raw_to_unified/converter/waymo/waymo_protos/vector.proto\x12\x12waymo.open_dataset\" \n\x08Vector2d\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\"+\n\x08Vector3d\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01')
)




_VECTOR2D = _descriptor.Descriptor(
  name='Vector2d',
  full_name='waymo.open_dataset.Vector2d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='waymo.open_dataset.Vector2d.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='waymo.open_dataset.Vector2d.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=92,
  serialized_end=124,
)


_VECTOR3D = _descriptor.Descriptor(
  name='Vector3d',
  full_name='waymo.open_dataset.Vector3d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='waymo.open_dataset.Vector3d.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='waymo.open_dataset.Vector3d.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='waymo.open_dataset.Vector3d.z', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=126,
  serialized_end=169,
)

DESCRIPTOR.message_types_by_name['Vector2d'] = _VECTOR2D
DESCRIPTOR.message_types_by_name['Vector3d'] = _VECTOR3D
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Vector2d = _reflection.GeneratedProtocolMessageType('Vector2d', (_message.Message,), dict(
  DESCRIPTOR = _VECTOR2D,
  __module__ = 'scenariomax.raw_to_unified.converter.waymo.waymo_protos.vector_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Vector2d)
  ))
_sym_db.RegisterMessage(Vector2d)

Vector3d = _reflection.GeneratedProtocolMessageType('Vector3d', (_message.Message,), dict(
  DESCRIPTOR = _VECTOR3D,
  __module__ = 'scenariomax.raw_to_unified.converter.waymo.waymo_protos.vector_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Vector3d)
  ))
_sym_db.RegisterMessage(Vector3d)


# @@protoc_insertion_point(module_scope)
