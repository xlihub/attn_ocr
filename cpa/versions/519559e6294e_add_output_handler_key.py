"""add output_handler key

Revision ID: 519559e6294e
Revises: 4ccab8b673b7
Create Date: 2022-06-02 15:09:26.675207

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '519559e6294e'
down_revision = '4ccab8b673b7'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('patterns', sa.Column('output_handle', sa.JSON(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('patterns', 'output_handle')
    # ### end Alembic commands ###
